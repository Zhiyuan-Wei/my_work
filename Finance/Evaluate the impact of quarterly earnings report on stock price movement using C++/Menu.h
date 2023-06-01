#include "SampledGroup.h"
#include "StockInfo.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include "calender.h"
#include "DataLoader.h"
#include "Bootstrap.h"
#include <thread>

using namespace std;
using namespace fre;

typedef vector<double> Vector;
typedef vector<Vector> Matrix;

class menu{
public:
    //data members
    int N;
    bool run_flag=true;
    static calender cal;
    static MarketInfo etf;
    SampledGroup Beat_ALL;
    SampledGroup Meet_ALL;
    SampledGroup Miss_ALL;

    StockGroup sg;
    vector<vector<string>> ThreeGroups;
    // vector<vector<string>> random_samples;
    vector<vector<CalculationResult>> result;
    vector<SampledGroup*> sg_vec;
    
    //constructor
    menu(){init_menu();}
    
    //member functions
    void Option_1();
    void Option_2();
    void Option_3();
    void Option_4();

    void  print_Options();
    int OptionSelect(){
        int option_num = 0;
        run_flag=true;
        cout<<"Please Enter Number 1-5 and Press Return: ";
        while (cin >> option_num){
            // cin>>option_num;
            switch (option_num) {
                case 1:
                    Option_1();
                    break;
                case 2:
                    Option_2();
                    break;
                case 3:
                    Option_3();
                    break;
                case 4:
                    Option_4();
                    break;
                case 5:
                    run_flag=false;
                    break;
                default:
                    cout << "There is No Data for the Symbol!" << endl;
                    break;
            }   
            if (run_flag) cout<<"Please Enter Number 1-5 and Press Return: ";
            else break;
        }

        
    }
    
    void init_menu(){
        print_Options();
        OptionSelect();
    }
};
