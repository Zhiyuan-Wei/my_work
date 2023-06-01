
#include "DataLoader.h"
#include "StockInfo.h"
//
void StockInfo::populateDates(int N, const calender &cal) {
    StartDate = cal.getDateAfter(EarningsDate, -N);
    EndDate = cal.getDateAfter(EarningsDate, N);
}

void StockInfo::printPriceInfo() {
    // print the first 10 elements
    int i = 0;
    for (auto it = PricingInfo.begin(); it != PricingInfo.end(); ++it) {
        cout << it->first << " " << it->second << endl;
        i++;
        if (i == 10) break;
    }
}

// void StockInfo::printCumReturnInfo() {
//     // print the first 10 elements
//     int i = 0;
//     for (auto it = CumReturnInfo.begin(); it != CumReturnInfo.end(); ++it) {
//         cout << it->first << " " << it->second << endl;
//         i++;
//         if (i == 10) break;
//     }
// }

void StockInfo::populatePricingInfo(CURL * handle){
//    int flag = GetSingleStockPrice( StartDate, EndDate, Symbol, PricingInfo, CumReturnInfo, handle);
    int flag = GetSingleStockPrice( StartDate, EndDate, Symbol, PricingInfo, handle);
}

void MarketInfo::populateDates(string start, string end)
{
    StartDate=start;
    EndDate=end;
}

map<string, double> mergeMaps(const map<string, double>& stockMap, const map<string, double>& marketMap) {
    map<string, double> mergedMap;
    for (const auto& kvp : stockMap) {
        string key = kvp.first;
        double value = marketMap.at(key);
        mergedMap[key] = value;
    }
    return mergedMap;
}   

map<string, double> MarketInfo::getPrice(map<string, double> stock_price)
{
    return mergeMaps(stock_price, PricingInfo);
}

// map<string, double> MarketInfo::getCumReturn(map<string, double> stock_price)
// {
//     return mergeMaps(stock_price, CumReturnInfo);
// }

void MarketInfo::populatePricingInfo(CURL * handle){
//    int flag = GetSingleStockPrice(StartDate, EndDate, Symbol, PricingInfo, CumReturnInfo, handle);
    int flag = GetSingleStockPrice(StartDate, EndDate, Symbol, PricingInfo, handle);
}

