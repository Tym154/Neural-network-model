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
vector<vector<int>> data;

//reading the data from a csv file and storing it into a 2D array
void reading_data(){
    ifstream CSVdata("../mnist_train.csv");
    string line;
    string temp;
    int count = 0;

    while(getline(CSVdata, line)){
        stringstream row(line);
        getline(CSVdata, temp, ',');
        label.push_back(stoi(temp));

        vector<int> tempVector;
        
    }

    cout << count;
}



int main(){
    reading_data();

    return 0;
}