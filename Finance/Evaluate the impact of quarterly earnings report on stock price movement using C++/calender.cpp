#include "calender.h"

void calender::printTradingDates() const {
    for (int i = 0; i < tradingDates.size(); i++) {
        cout << tradingDates[i] << endl;
    }
}

string calender::getDateAfter(string date, unsigned int N) const
{
    int index = 0;
    auto it = std::find(tradingDates.begin(), tradingDates.end(), date);
    if (it != tradingDates.end()) {
        index = std::distance(tradingDates.begin(), it);
    }
    else {
        cout << "The date is not in the trading dates." <<date << endl;
        return "";
    }
    index += N;
    return tradingDates[index];
}
