#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include "StockInfo.h"
#include "SampledGroup.h"
#include "StockInfo.h"
#include "Utils.hpp"
#include "Utils_template.h"
#include <vector>
#include <random>
using namespace std;

vector<string> random_sample_group(const vector<string>& Group);

class Bootstrap {
protected:
    int n_iter;
    int n_group;
    int n_days;
    MarketInfo etf;
    // first dimention represent Beat Meet Miss, second dimension represents average AAR, average CAAR, std AAR, std CAAR
    vector<vector<CalculationResult>> calculated_results; 
public:
    Bootstrap(int n_iter, int n_group, int n_days, MarketInfo etf_): n_iter(n_iter), n_group(n_group), n_days(n_days), etf(etf_) {}
    int aggregate_N_iter(vector<SampledGroup*> sg_vec, vector<vector<string>>& ThreeGroups);
    vector<vector<CalculationResult>> get_result() {return calculated_results;}
    void setETF(MarketInfo etf_) { etf = etf_; }
};


