#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include "calender.h"
using namespace std;
// abstract class AssetInfo
class AssetInfo {
protected:
    string StartDate;
    string EndDate;
    string Symbol;
    map<string, double> PricingInfo; //map date:price, time series
//    map<string, double> CumReturnInfo;
public:
    AssetInfo():Symbol("UnDefined"){}
    AssetInfo(string symbol_) : Symbol(symbol_) {}
    virtual ~AssetInfo() {}
    // getter
    string getSymbol() const { return Symbol; }
    string getStartDate() const { return StartDate; }
    string getEndDate() const { return EndDate; }
    map<string, double> getPricingInfo() const { return PricingInfo; }
//    map<string, double> getCumReturnInfo() const { return CumReturnInfo; }
};
    
// StockInfo class with earnings info (every stock besides etf)
class StockInfo  : public AssetInfo{
private:
//    string Symbol;
    string EarningsDate;
    string PeriodEndDate;
    double EstimatedEPS;
    double ReportedEPS;
    double EarningsSurprise;
    double SurprisePercent; // sort by this
public:
    // constructor
    StockInfo(){}
    StockInfo(string symbol_, string earningsDate_, string periodEndDate_,
              double estimatedEPS_, double reportedEPS_, double earningsSurprise_,
              double surprisePercent_)
              :AssetInfo(symbol_), EarningsDate(earningsDate_),PeriodEndDate(periodEndDate_),
              EstimatedEPS(estimatedEPS_), ReportedEPS(reportedEPS_),
              EarningsSurprise(earningsSurprise_), SurprisePercent(surprisePercent_) {}
    // destructor
    virtual ~StockInfo() {}
    // getter
    string getEarningsDate() const { return EarningsDate; }
    string getPeriodEndDate() const { return PeriodEndDate; }
    double getEstimatedEPS() const { return EstimatedEPS; }
    double getReportedEPS() const{ return ReportedEPS; }
    double getEarningsSurprise() const{ return EarningsSurprise; }
    double getSurprisePercent() const{ return SurprisePercent; }
    // setter
    void populateDates(int N, const calender &cal);//calculate StartDate & EndDate with EarningsDate
    void populatePricingInfo(CURL * handle);//get map<string, double> pricingInfo using StartDate,EndDate,Symbol
    void printPriceInfo();
//    void printCumReturnInfo();
    // map<string, double> getPriceInfo() {return PricingInfo;}
};


class MarketInfo : public AssetInfo {
public:
    MarketInfo() : AssetInfo() {}
    MarketInfo(string symbol_) :AssetInfo(symbol_) {}
    // destructor
    virtual ~MarketInfo() {}
    // void setPricingInfo();//get map<string, double> pricingInfo as well as StartDate and EndDate for all available market data
    map<string, double> getPrice(map<string, double> stock_price); //return market priceinfo map with same time scale as stock_price
//    map<string, double> getCumReturn(map<string, double> stock_price);
    map<string, double> getPrice(){return PricingInfo;} ////return market priceinfo map from startdate to end date
//    map<string, double> getCumReturn(){return CumReturnInfo;}
    void populateDates(string start, string end);
    // void printInfo();
    void populatePricingInfo(CURL * handle);
    
};




