#pragma once
#include "StockInfo.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <algorithm>
#include <random>
#include "Utils.hpp"
#include "Utils_template.h"


using namespace std;
//
namespace fre {
    bool cmp_val(const std::pair<std::string, StockInfo>& a, const std::pair<std::string, StockInfo>& b);
    // vector<vector<string>> random_sample_from_groups(const vector<vector<string>>& Groups);
    
}

using namespace fre;

typedef vector<double> Vector;
typedef vector<Vector> Matrix;
//
class StockGroup{
protected:
    string FileName;
    map<string, StockInfo> StockMap;
    vector<string> Symbols;
    MarketInfo etf;
public:
    // constructor
    StockGroup();
    StockGroup(string fileName_) : FileName(fileName_){}
    // destructor
    virtual ~StockGroup() {};
    // getter
    vector<string> GetSymbols() const { return Symbols; }
    string getFileName() { return FileName; }
    StockInfo getStockInfo(string symbol_) const { return StockMap.at(symbol_); }
    // setter
    void populateStockMap();
    void setETF(MarketInfo etf_) {etf = etf_; }
    // print
    void printInfo();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // sort
    void sortStockMapBySurprisePercent();
    // populate the price info of all stocks in the group
    // divide the stock into three groups
    vector<vector<string>> dividestock();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

class SampledGroup : public StockGroup
{
private:
    Vector AAR;
    Vector CAAR;
    MarketInfo etf;
    int n_days;

public:
    // constructor
    SampledGroup() {}
    SampledGroup(vector<string> Symbols_, const StockGroup &stockGroup, MarketInfo etf_, int n_days_); // constructor
    SampledGroup(vector<string> Symbols_, const SampledGroup& sg, MarketInfo etf_, int n_days_); // constructor
    // update the data member Symbols
    void update(vector<string> Symbols_, const StockGroup &stockGroup, MarketInfo etf_, int n_days_);

    // populate Dates
    void populateDates(int N, const calender &cal);
    // populate Price
    void populatePricingInfo(CURL * handle);
    // print
    void printInfo() const;
    void calculateAAR();
    void calculateCAAR();
    void printndats() const{cout << "n_days: " << n_days << endl;}

    Vector getAAR() {return AAR;}
    Vector getCAAR() {return CAAR;}
    CalculationResult calculation();
};
static void populate_pricing_info(SampledGroup& sg, CURLM* multi_handle)
{
    // create a new easy handle
    CURL* handle = curl_easy_init();

    // populate the pricing info for the SampledGroup object using the easy handle
    sg.populatePricingInfo(handle);

    // add the handle to the multi handle
    curl_multi_add_handle(multi_handle, handle);
}
static vector<vector<string>> random_sample_from_groups(const vector<vector<string>>& Groups)
{
    // create three vectors to store the random samples
    vector<string> group1_sample(80);
    vector<string> group2_sample(80);
    vector<string> group3_sample(80);

    vector<string> copy1 = Groups[0];
    vector<string> copy2 = Groups[1];
    vector<string> copy3 = Groups[2];

    // create a random number generator
    default_random_engine rng{random_device{}()};

    // shuffle the three vectors
    shuffle(copy1.begin(), copy1.end(), rng);
    shuffle(copy2.begin(), copy2.end(), rng);
    shuffle(copy3.begin(), copy3.end(), rng);

    // copy the first 80 elements of the shuffled vectors to the three sample vectors
    copy_n(copy1.begin(), 80, group1_sample.begin());
    copy_n(copy2.begin(), 80, group2_sample.begin());
    copy_n(copy3.begin(), 80, group3_sample.begin());

    // group1_sample=randomStockSelection(ThreeGroups[0]);
    // group2_sample=randomStockSelection(ThreeGroups[1]);
    // group3_sample=randomStockSelection(ThreeGroups[2]);

    // return the three sample vectors
    return {group1_sample, group2_sample, group3_sample};
}

