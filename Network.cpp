#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <bits/stdc++.h>

using namespace std;

vector<int> label;
vector<vector<int>> data_from_csv;
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

class Node{
    public:
    vector<double> weights; //weights on inputs
    vector<double> inputs;  //inputs to the node
    double bias;            //just... bias
    double output;          //activation
    double value;           //weighted input

    //constructor for the node class
    Node(int input_size){
        random_device rd; //obtain a random number from hardware
        mt19937 gen(rd()); //seed the generator
        uniform_real_distribution<> distr(-1.0, 1.0); //random double between -1 and 1

        for(int i = 0; i < input_size; i++){ //setting random number between 1 and -1 for weights and a bias
            weights.push_back(distr(gen));
        }
        bias = (distr(gen));

        value = 0.0; //reseting the values to 0 so they dont acumulate when learning
        output = 0.0;
    }

    //using leaky relu for the activation function in hidden layers
    double activation(double x){
        double alfa = 0.01;
        return (x > 0) ? x : x * alfa;
    }

    //forward function
    double forward(vector<double> x){ 
        inputs = x; //storing the inputs for later
        
        for(int i = 0; i < x.size(); i++){ //summing up all the inputs * weight on the input
            value += x[i] * weights[i];
        }

        value += bias; //adding the bias
        output = activation(value); //getting the activation
        return output;
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
        layers.emplace_back(Layer(num_nodes[0], num_nodes[0])); //adding the input layer

        for(int i = 1; i < num_nodes.size(); i++){
            layers.emplace_back(Layer(num_nodes[i], num_nodes[i-1])); //adding the hidden layers and output layer
        }
    }


    //Cost function (MSE)
    void cost(){
        double cost = 0;

        for(int i = 0; i < layers.back().nodes.size(); i++){ //iterating through the output layer
            if(i != label[cycle]){ 
                cost += pow(layers.back().nodes[i].output, 2);
            }

            else{
                cost += pow(layers.back().nodes[i].output - label[cycle], 2);
            }
        }

        cycle++; //counting the cycles

        cost = cost * (1/layers.back().nodes.size());

        layers.back().cost = cost;
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

    for(vector<int> line : data_from_csv){
        cout << "\n\n\n";

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
    reading_data();
    return 0;
}


//help, really