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
#include "Menu.h"


using namespace std;
using namespace fre;

typedef vector<double> Vector;
typedef vector<Vector> Matrix;

template<class T>
void formatted_display(vector<T>& vec, int num_per_line, int width, int precision) {
    cout << setiosflags(ios::fixed) << setprecision(precision);
    for (int i = 0; i < vec.size(); ++i) {
        cout << setw(width) << vec[i];
        if ((i+1) % num_per_line == 0) cout << endl;
    }
    if ((vec.size() + 1) % num_per_line != 0) cout << endl;
}


MarketInfo menu::etf("IWV");
calender menu::cal;
void menu::print_Options(){
        cout<<endl;
        cout<<"1 - Enter N to retrieve 2N+1 days of historical price data for all stocks "<<endl;
        cout<<"2 - Pull information for one stock from one group "<<endl;
        cout<<"3 - Show AAR, AAR-STD, CAAR and CAAR-STD for one group "<<endl;
        cout<<"4 - Show CAAR for all 3 groups "<<endl;
        cout<<"5 - Exit "<<endl;
        cout << endl;
}
    
void menu::Option_1(){
    if (sg_vec.size()) {
        // clear all the data member
        sg_vec.clear();
        ThreeGroups.clear();

        Beat_ALL = SampledGroup();
        Meet_ALL = SampledGroup();
        Miss_ALL = SampledGroup();

    }
    cout<<"Enter N (N >= 60 & N<=90): ";
    cin>>N;
        // test 90 >= N >= 60
    if (N < 60 || N>90 ){cout << "N must >= 60 and <=90!" << endl;}
    else{
        /*
        retreve data here
        */
        CURL* handle;
        curl_global_init(CURL_GLOBAL_ALL);
        handle = curl_easy_init();
        if (!handle) {
            fprintf(stderr, "Curl init failed!\n");
            return;
        }

        string start="2017-01-03";
        string end="2023-05-01";
    
        // get the etf
        etf.populateDates(start, end);
        etf.populatePricingInfo(handle);
        curl_easy_cleanup(handle);
        curl_global_cleanup();

        // populate the trading dates
        vector<string> trading_dates;
        map<string, double> etf_info = etf.getPricingInfo();
        for (auto it = etf_info.begin(); it != etf_info.end(); ++it) {
            trading_dates.push_back(it->first);
        }
        cal.setTradingDates(trading_dates);

        // get the stockgroup
        string EarningsFileName = "Russell3000EarningsAnnouncements.csv";
        StockGroup temp(EarningsFileName);
        sg = temp;

        // store 3000 stocks' earnings info
        sg.populateStockMap();
        // sort
        sg.sortStockMapBySurprisePercent();
        ThreeGroups = sg.dividestock();

        // populate the price info of all stocks in the group using multithreading
//        static SampledGroup Beat_ALL(ThreeGroups[0], sg, etf, N); // num_all / 3
//        static SampledGroup Meet_ALL(ThreeGroups[1], sg, etf, N);
//        static SampledGroup Miss_ALL(ThreeGroups[2], sg, etf, N);
        Beat_ALL.update(ThreeGroups[0], sg, etf, N);
        Meet_ALL.update(ThreeGroups[1], sg, etf, N);
        Miss_ALL.update(ThreeGroups[2], sg, etf, N);

        Beat_ALL.printndats();
        // populate begin date and end date
        Beat_ALL.populateDates(N, cal);
        Meet_ALL.populateDates(N, cal);
        Miss_ALL.populateDates(N, cal);

        // initialize libcurl
        curl_global_init(CURL_GLOBAL_ALL);
        // create a multi handle
        CURLM* multi_handle = curl_multi_init();
        vector<thread> threads;
        // generate the handels
        curl_global_init(CURL_GLOBAL_ALL);
        threads.emplace_back(populate_pricing_info, std::ref(Beat_ALL), multi_handle); //populate Beat_all price info
        threads.emplace_back(populate_pricing_info, std::ref(Meet_ALL), multi_handle); //populate Meet_all price info
        threads.emplace_back(populate_pricing_info, std::ref(Miss_ALL), multi_handle); //populate Miss_all price info
    
        // perform the transfers
        int still_running;
        do {
            curl_multi_perform(multi_handle, &still_running);
        } while (still_running);
    
        // wait for the threads to finish
        for (auto& t : threads) {
            t.join();
        }
    
        // output message after all threads complete
        std::cout << "Pricing information population complete" << endl;
    
        // cleanup
        curl_multi_cleanup(multi_handle);
        curl_global_cleanup();
    
        // Sample from the Beat_ALL, Meet_ALL, Miss_ALL
        ThreeGroups[0] = Beat_ALL.GetSymbols();
        ThreeGroups[1] = Meet_ALL.GetSymbols();
        ThreeGroups[2] = Miss_ALL.GetSymbols();

        // calculate the AAR and CAAR
        sg_vec = vector<SampledGroup*>{&Beat_ALL, &Meet_ALL, &Miss_ALL};
        Bootstrap three_groups(40, 3, N, etf);
        three_groups.aggregate_N_iter(sg_vec, ThreeGroups);
        result = three_groups.get_result();
        cout << "Calculation finished" << endl;
        cout<<"== Option 1 completed =="<< endl << endl;
    }
}

void menu::Option_2(){
    if (! sg_vec.size()) {
        cout << "Please run Option 1 first" << endl;
        return;
    }
    string symbol;
    vector<string> group_names = {"Beat", "Meet", "Miss"};
    bool hasSymbol = false;
    int ix = 0;
    
    // test has data for the symbol
    if (!(cin)) cin.clear();
    cout<<"Enter Symbol for Detailed Information: ";
    cin >> symbol;
    cout << symbol << "/" << endl;
        
    for (ix = 0; ix < 3; ++ix) {
        if (find(ThreeGroups[ix].begin(), ThreeGroups[ix].end(), symbol) != ThreeGroups[ix].end()) {
            break;
        }
    }    
        
    if (ix != 3) {
        hasSymbol = true;
    }
    else {
        cout << "There is No Data for the Symbol!" << endl;
        return;
    }
    cout << endl;
    // print stock info
    cout << "Ticker:                " << sg_vec[ix]->getStockInfo(symbol).getSymbol() << endl;
    cout << "Earnings Call Date:    " << sg_vec[ix]->getStockInfo(symbol).getEarningsDate() << endl;
    cout << "Period Ending:         " << sg_vec[ix]->getStockInfo(symbol).getPeriodEndDate() << endl;
    cout << "Reported EPS:          " << sg_vec[ix]->getStockInfo(symbol).getReportedEPS() << endl;
    cout << "Estimated EPS:         " << sg_vec[ix]->getStockInfo(symbol).getEstimatedEPS() << endl;
    cout << "Surprise:              " << sg_vec[ix]->getStockInfo(symbol).getEarningsSurprise() << endl;
    cout << "Surprise%:             " << sg_vec[ix]->getStockInfo(symbol).getSurprisePercent() << endl;
    cout << endl << "Group:                 " << group_names[ix] << endl;

    cout << endl;
    
    Vector price = map_values_to_vec(sg_vec[ix]->getStockInfo(symbol).getPricingInfo());
    vector<string> dates = map_keys_to_vec(sg_vec[ix]->getStockInfo(symbol).getPricingInfo());
    double base = price[0];
    
    // print out dates
    cout << "Daily Price Dates:" << endl;
    formatted_display(dates, 6, 14, 0); cout << endl;
    // print out prices
    cout << "Daily Prices: " << endl;
    formatted_display(price, 10, 10, 3); cout << endl;
    // print out cumulative daily returns
    price = price / base - 1.0; price = Vector(price.begin() + 1, price.end());
    cout << "Cumulative Daily Returns" << endl;
    formatted_display(price, 10, 10, 4); cout << endl;
    
    cout<<"== Option 2 completed =="<< endl << endl;
}

void menu::Option_3() {
    if (! sg_vec.size()) {
        cout << "Please run Option 1 first" << endl;
        return;
    }
    string grp_str;
    cout << "Enter Group (Beat, Meet, Miss) for Detailed Information: ";
    if (!(cin)) cin.clear();
    cin >> grp_str;
    
    map<string, int> group_dict = {{"Beat", 0}, {"Meet", 1}, {"Miss", 2}};
    if (! group_dict.count(grp_str)) {
        cout << "Group name incorrect (not in Beat Meet Miss)!" << endl;
        return;
    }
    
    int grp_ix = group_dict[grp_str];
    
    // Avg AAR
    cout << "Avg AAR for Group " << grp_str << endl;
    formatted_display(result[grp_ix][0].AAR, 10, 10, 4); cout << endl;
    // AAR STD
    cout << "AAR STD for Group " << grp_str << endl;
    formatted_display(result[grp_ix][1].AAR, 10, 10, 4); cout << endl;
    // AVG CAAR
    cout << "Avg CAAR for Group " << grp_str << endl;
    formatted_display(result[grp_ix][0].CAAR, 10, 10, 4); cout << endl;
    // CAAR STD
    cout << "CAAR STD for Group " << grp_str << endl;
    formatted_display(result[grp_ix][1].CAAR, 10, 10, 4); cout << endl;
    
    cout<<"== Option 3 completed =="<< endl << endl;
}

void menu::Option_4(){
    if (! sg_vec.size()) {
        cout << "Please run Option 1 first" << endl;
        return;
    }
    plotResults(result,N);
    cout<<"== Option 4 completed =="<< endl << endl;
}

