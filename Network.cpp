#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <random>

using namespace std;

vector<int> label; // labels
vector<vector<int>> expected_outputs;
vector<vector<int>> data_from_csv; //data converted from a csv file
int cycle = 0;

//reading the data from a csv file and storing it into a 2D array
void reading_data(){
    ifstream CSVdata("../mnist_train.csv");
    string line;
    string temp;

    while(getline(CSVdata, line)){
        stringstream row(line);

        getline(row, temp, ',');
        label.push_back(stoi(temp));

        vector<int> tempVector;

        while(getline(row, temp, ',')){
            tempVector.push_back(stoi(temp));
        }

        data_from_csv.push_back(tempVector);
    }
}

//preparing expected_output from label
void expect(){
    for(int i = 0; i < label.size(); i++){
        vector<int> temp(10, 0);

        for(int j = 0; j < 10; j++){
            if(label[i] == j){
                temp[j] = 1;
                break;
            }
        }

        expected_outputs.push_back(temp);
    }
}


class Node{
    public:
    vector<double> weights; //weights on inputs
    vector<double> inputs;  //inputs to the node
    double bias;            //just... bias
    double output;          //activation
    double value;           //weighted input
    double delta;           //delta for backpropagation

    //constructor for the node class
    Node(int input_size){
        random_device rd; //obtain a random number from hardware
        mt19937 gen(rd()); //seed the generator
        uniform_real_distribution<> distr(-1.0, 1.0); //random double between -1 and 1

        for(int i = 0; i < input_size; i++){ //setting random number between 1 and -1 for weights and a bias
            weights.push_back(distr(gen));
        }
        bias = distr(gen);

        value = 0.0; //reseting the values to 0 so they dont acumulate when learning
        output = 0.0;
    }

    //using leaky relu for the activation function in hidden layers
    double activation(double x){
        double alfa = 0.01;
        return (x > 0) ? x : x * alfa;
    }

    //derivative of the leaky relu function;
    double activation_derivative(double x){
        double alfa = 0.01;
        return (x > 0) ? 1 : alfa;
    }
};



class Layer{
    public:
    vector<Node> nodes;
    double cost;
    
    //layer constructor (just pushing back new node)
    Layer(int num_nodes, int input_size){
        for(int i = 0; i < num_nodes; i++){
            nodes.emplace_back(Node(input_size)); 
        }
    }

};



class Network{
    public:
    vector<Layer> layers;

    //Network constructor
    Network(vector<int> num_nodes){
        layers.emplace_back(Layer(num_nodes[0], 0)); //adding the input layer

        for(int i = 1; i < num_nodes.size(); i++){
            layers.emplace_back(Layer(num_nodes[i], num_nodes[i-1])); //adding the hidden layers and output layer
        }
    }


    //Cost function (MSE)
    void cost(){
        double cost = 0;

        for(int i = 0; i < layers.back().nodes.size(); i++){
            cost += pow(layers.back().nodes[i].output - expected_outputs[cycle][i], 2);
        }

        cycle++; //counting the cycles

        cost = cost / layers.back().nodes.size();

        layers.back().cost = cost;
    }

    //forward function
    void forward_propagation(){
        //iterating throught layers
        for(int i = 0; i < layers[0].nodes.size(); i++){
            Node &node = layers[0].nodes[i];
            node.value = data_from_csv[cycle][i];
            node.output = node.activation(node.output);
        }

        for(int i = 1; i < layers.size(); i++){
            
            //iterating throught nodes
            for(int j = 0; j < layers[i].nodes.size(); j++){
                Node &node = layers[i].nodes[j];
                node.value = 0;

                //iterating throught weights connected to node
                for(int k = 0; k < node.weights.size(); k++){
                    node.value += node.weights[k] * layers[i-1].nodes[k].output; //adding up all the weighted inputs
                }

                node.value += node.bias;
                node.output = node.activation(node.value);
            }
        }
    }

    //backpropagation
    void backpropagate(double learning_rate, vector<int> expected_output){

        //output layer deltas
        for(int i = 0; i < layers.back().nodes.size(); i++){
            Node &node = layers.back().nodes[i];
            double error = node.output - expected_output[i];
            node.delta = error * node.activation_derivative(node.value);
        }

        //hidden layer deltas
        for(int i = layers.size() - 2; i >= 0; i--){
            for(int j = 0; j < layers[i].nodes.size(); j++){
                Node &node = layers[i].nodes[j];
                double error = 0;
                
                for(int k = 0; k < layers[i+1].nodes.size(); k++){
                    error = layers[i+1].nodes[k].delta * layers[i+1].nodes[k].weights[j];
                }

                node.delta = error * node.activation_derivative(node.value);
            }
        }

        //updating weights and biases
        for(int i = 1; i < layers.size(); i++){
            for(int j = 0; j < layers[i].nodes.size(); j++){
                Node &node = layers[i].nodes[j];

                for(int k = 0; k < node.weights.size(); k++){
                    node.weights[k] -= node.delta * learning_rate * layers[i-1].nodes[k].output;
                }

                node.bias -= learning_rate * node.delta;
            }
        }
    }

    /////////////////testing functions//////////////////
    void Display_weights(){ 
        for(int i = 0; i < layers.size(); i++){ //iterating through layers

            cout << "Layer: " << i << " ";
            if(i == layers.size()-1) cout << "(output)"; //if the layer is last it is the output layer so i will cout that
            cout << "\n";

            for(int j = 0; j < layers[i].nodes.size(); j++){ //iterating through nodes
                int count = 0; //reseting the counter of weights

                for(int k = 0; k < layers[i].nodes[j].weights.size(); k++){ //iterating through weights
                    cout << layers[i].nodes[j].weights[k] << "  ";
                    count++; //counting the weights in a layer to check how many are connected to the node
                }

                cout << "\nNumber of weights: "<< count <<"\n\n";
            }

            cout << "\n\n\n\n"; 

        }
    }

    void Display_bias(){
        for(int i = 0; i < layers.size(); i++){ //iterating through layers

            int count = 0; //reseting the counter of biases

            cout << "Layer: " << i << " ";

            if(i == layers.size()-1) cout << "(output)"; //if the layer is last it is the output layer so i will cout that
            cout << "\n";

            for(int j = 0; j < layers[i].nodes.size(); j++){ //iterating through nodes
                cout << layers[i].nodes[j].bias << " ";
                count++;
            }

            cout << "\nNumber of biases (and nodes):"<< count <<"\n\n"; 
        }
    }
    ////////////////////////////////////////////////////
};


////////////////////testing functions////////////////////
void Display_data(){

    int count_of_elements = 0;
    int count_of_lines = 0;

    for(int i = 0; i < data_from_csv.size(); i++){
        vector<int> line = data_from_csv[i];

        cout << "\n\n\nLabel for this data: " << label[i] << "\n\n";

        for(int num : line){
            cout << " " << num;
            count_of_elements++;
        }

        cout << "\nNUmber of values" << count_of_elements;
        count_of_elements = 0;
        count_of_lines++;
    }

    cout << "\n\nCount of lines:" << count_of_lines;
}
//////////////////////////////////////////////////////////

//Just testing the functions inside the main
int main() {
    int count;
    double learning_rate = 0.00001;
    Network network({784, 100, 100, 100, 10});

    reading_data();
    expect();

    for(int i = 0; i < 10000; i++){
    network.forward_propagation();
    network.cost();
    cout << "\n \n" << network.layers.back().cost;
    network.backpropagate(learning_rate, expected_outputs[i]);
    }
}   


//help, really