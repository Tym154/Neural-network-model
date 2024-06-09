#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <bits/stdc++.h>

using namespace std;

int label;

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
};


//Just testing the functions inside the main
int main() {
    return 0;
}


//help