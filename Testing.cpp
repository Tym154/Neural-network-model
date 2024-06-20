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



int main(){
    reading_data();

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

    return 0;
}