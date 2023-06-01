#pragma once
#include "DataLoader.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <algorithm>
using namespace std;

class calender {
private:
    vector<string> tradingDates;
public:
    // constructor
    calender() {}
    // getter
    vector<string> getTradingDates() { return tradingDates; }
    // setter
    void setTradingDates(vector<string> tradingDates_) { tradingDates = tradingDates_; }
    // print
    void printTradingDates() const;
    // given a date and a number of days, return the date after the given number of days
    string getDateAfter(string date, unsigned int N) const;
};
