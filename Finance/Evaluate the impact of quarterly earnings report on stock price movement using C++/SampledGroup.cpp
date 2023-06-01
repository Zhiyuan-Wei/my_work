//
// Created by 彭琪 on 4/27/23.
//
#include "SampledGroup.h"
//include "Utils.hpp"
//#include "Utils_template.cpp"
using namespace fre;

// Transform the date format from "21-Nov-2022" to "2022-11-21"
string transformDate(string date) {
    map<string, string> month_map;
    month_map["Jan"]="01";
    month_map["Feb"]="02";
    month_map["Mar"]="03";
    month_map["Apr"]="04";
    month_map["May"]="05";
    month_map["Jun"]="06";
    month_map["Jul"]="07";
    month_map["Aug"]="08";
    month_map["Sep"]="09";
    month_map["Oct"]="10";
    month_map["Nov"]="11";
    month_map["Dec"]="12";
    string year = date.substr(7, 4);
    string month = month_map[date.substr(3, 3)];
    string day = date.substr(0, 2);
    return year + "-" + month + "-" + day;
}

// constructor
StockGroup::StockGroup() {
    FileName = "Russell3000EarningsAnnouncements.csv";
}

void StockGroup::populateStockMap()
{
    // Set up the stock map
    string line;
    ifstream fin;
    fin.open(FileName, ios::in);
    string symbol, earningsDate, periodEndDate;
    double estimate, reported, surprise, surprisePercent;
    getline(fin, line); // skip the first line
    //get stock symbol, EA date, surprise%
    while (getline(fin, line))
    {
        symbol = line.substr(0, line.find_first_of(','));
        Symbols.push_back(symbol);
        line.erase(0, line.find_first_of(',')+1);

        earningsDate = line.substr(0, line.find_first_of(',')); // date is in the format of "21-Nov-2022"
        earningsDate = transformDate(earningsDate); // date is in the format of "2022-11-21"
        line.erase(0, line.find_first_of(',')+1);

        periodEndDate = line.substr(0, line.find_first_of(','));
        line.erase(0, line.find_first_of(',')+1);

        estimate = stod(line.substr(0, line.find_first_of(',')));
        line.erase(0, line.find_first_of(',')+1);

        reported = stod(line.substr(0, line.find_first_of(',')));
        line.erase(0, line.find_first_of(',')+1);

        surprise = stod(line.substr(0, line.find_first_of(',')));
        line.erase(0, line.find_first_of(',')+1);

        surprisePercent = stod(line.substr(0, line.find_first_of(',')));
        line.erase(0, line.find_first_of(',')+1);

        if (symbol == "LICY")
        {
            earningsDate = "2023-01-30";
        }
        // create a stock info
        StockInfo stockInfo(symbol, earningsDate, periodEndDate, estimate, reported, surprise, surprisePercent);
        // stockInfo.populateDates(N, cal);
        // stockInfo.populatePricingInfo();
        // add the stock info to the map
        // initialize the map
//        StockMap = map<string, StockInfo>();
        StockMap.insert(pair<string, StockInfo>(symbol, stockInfo));
    }
    fin.close();
}


// print the stock map
void StockGroup::printInfo() {
    for (auto it = StockMap.begin(); it != StockMap.end(); it++) {
        cout << it->first << " " << it->second.getEarningsDate() << " " << it->second.getSurprisePercent() << endl;
    }
}

// sample group constructor
SampledGroup::SampledGroup(vector<string> Symbols_, const StockGroup &stockGroup, MarketInfo etf_, int n_days_)
{
    Symbols = Symbols_;
    etf = etf_;
    n_days = n_days_;
    // iterate through the symbols, and add the stock info to the map
    auto it = Symbols.begin();
    while (it != Symbols.end())
    {
        StockMap.insert(pair<string, StockInfo>(*it, stockGroup.getStockInfo(*it)));
        it++;
    }
}



SampledGroup::SampledGroup(vector<string> Symbols_, const SampledGroup& sg, MarketInfo etf_, int n_days_){
    Symbols = Symbols_;
    etf = sg.etf;
    n_days = sg.n_days;
    // iterate through the symbols, and add the stock info to the map
    auto it = Symbols.begin();
    while (it != Symbols.end())
    {
        StockMap.insert(pair<string, StockInfo>(*it, sg.getStockInfo(*it)));
        it++;
    }

}
void SampledGroup::update(vector<string> Symbols_, const StockGroup &stockGroup, MarketInfo etf_, int n_days_){
    Symbols = Symbols_;
    etf = etf_;
    n_days = n_days_;
    // iterate through the symbols, and add the stock info to the map
    auto it = Symbols.begin();
    while (it != Symbols.end())
    {
        StockMap.insert(pair<string, StockInfo>(*it, stockGroup.getStockInfo(*it)));
        it++;
    }
}
// populate Dates
void SampledGroup::populateDates(int N, const calender &cal) {
    // iterate through the stock map
    for (auto it = StockMap.begin(); it != StockMap.end(); it++) {
        // populate the dates
        it->second.populateDates(N, cal);
        // check if the dates are populated
        if (it->second.getStartDate() == "" || it->second.getEndDate() == "") {
            // if not, print the symbol
            cout << it->first << endl;
        }
    }
    cout << "Dates are populated" << endl;
}

// populate Price
void SampledGroup::populatePricingInfo(CURL * handle){
    vector<string> keys_to_erase;
    // iterate through the stock map
    for (auto it = StockMap.begin(); it != StockMap.end(); it++) {
        // populate the dates
        it->second.populatePricingInfo(handle);
        if (it->second.getPricingInfo().size() < 2 * n_days+1) {
            keys_to_erase.push_back(it->first);
            // get rid of the stock if the pricing info is not enough
        }

    }
    for (auto key : keys_to_erase) {
        StockMap.erase(key);
        // also the symbol
        auto it = find(Symbols.begin(), Symbols.end(), key);
        Symbols.erase(it);
        cout << key << " is erased" << endl;
    }
    cout << "Pricing info is populated" << endl;
}



// print the info of the sampled group
void SampledGroup::printInfo() const{
    for (auto it = StockMap.begin(); it != StockMap.end(); it++) {
        cout << it->first << " " << it->second.getEarningsDate() << " " << it->second.getSurprisePercent() << " " <<it->second.getStartDate() << " " << it->second.getEndDate() << endl;
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// sort
void StockGroup::sortStockMapBySurprisePercent() {
    // Create a vector of pairs of stock symbols and their corresponding StockInfo
    std::vector<std::pair<std::string, StockInfo>> vec(StockMap.begin(), StockMap.end());

    // Sort the vector by SurprisePercent
    std::sort(vec.begin(), vec.end(), [](const std::pair<std::string, StockInfo>& a, const std::pair<std::string, StockInfo>& b) {
        return a.second.getSurprisePercent() < b.second.getSurprisePercent();
    });

    // clear vector <string> Symbols
    Symbols.clear();
    // // populate the symbols by the sorted map
    for (const auto& p : vec) {
        Symbols.push_back(p.first);
    }


}

// 分组
vector<vector<string>> StockGroup::dividestock() {
    // 输出，后面方便检查
    // cout << Symbols.size() << endl;
    
    // 获取股票数量和每组股票数量
    int n = Symbols.size();
    int n_per_group = n / 3;
    
    // 分配股票符号到新的vector中
    vector<string> symbols1(Symbols.begin(), Symbols.begin() + n_per_group);
    vector<string> symbols2(Symbols.begin() + n_per_group, Symbols.begin() + 2*n_per_group);
    vector<string> symbols3(Symbols.begin() + 2*n_per_group, Symbols.end());
    
    vector<vector<string>> result;
    result.push_back(symbols3);
    result.push_back(symbols2);
    result.push_back(symbols1);
    return result;
}

// 随机抽取80个股票

// vector<vector<string>> random_sample_from_groups(const vector<vector<string>>& Groups)
// {
//     // 创建三个存储随机抽样结果的 vector
//     vector<string> group1_sample(80);
//     vector<string> group2_sample(80);
//     vector<string> group3_sample(80);
    
//     vector<string> copy1 = Groups[0];
//     vector<string> copy2 = Groups[1];
//     vector<string> copy3 = Groups[2];

//     // 随机数生成器
//     default_random_engine rng{random_device{}()};

//     // 随机打乱每组的元素
//     shuffle(copy1.begin(), copy1.end(), rng);
//     shuffle(copy2.begin(), copy2.end(), rng);
//     shuffle(copy3.begin(), copy3.end(), rng);

//     // 将每组前80个元素复制到相应的样本 vector 中
//     copy_n(copy1.begin(), 80, group1_sample.begin());
//     copy_n(copy2.begin(), 80, group2_sample.begin());
//     copy_n(copy3.begin(), 80, group3_sample.begin());
    
//     // group1_sample=randomStockSelection(ThreeGroups[0]);
//     // group2_sample=randomStockSelection(ThreeGroups[1]);
//     // group3_sample=randomStockSelection(ThreeGroups[2]);
    
//     // 返回随机抽样结果
//     return {group1_sample, group2_sample, group3_sample};
// }

// vector<string> randomStockSelection (vector<string> group)
//  {
//      vector<string> randomSampling;
//      //srand((unsigned)time(NULL));
//      int size=(int)group.size();
//      int nums[size];
//      for(int i=0;i<size;i++){
//          nums[i]=i;
//      }
//      // random_shuffle(first, last)： 将两个位置之间的数字打乱顺序
//      random_shuffle(nums,nums+size);
//      int number;
//      // 在打乱顺序后，取前80个数字，当 index
//      for(int i=0;i<80;i++){
//          number=nums[i];
//          // 压入对应随机index的stock
//          randomSampling.push_back(group[number]);
//      }
//      return randomSampling;
//  }

void SampledGroup::calculateAAR()
{
    Vector AR;
    vector<StockInfo> SampleVector = map_values_to_vec(StockMap);   // get vector of stock information
    int size = SampleVector.size();
    for (int i = 0; i<2*n_days; i++)
        AAR.push_back(0);
    for (int i = 0; i < size; i++) {
        // calculate abnormal return vector for each stock
        // get price of the 2N+1 days
        Vector PricingInfo = map_values_to_vec(SampleVector[i].getPricingInfo());
        // get return of the 2N days
        Vector ReturnVector = std::vector<double>(PricingInfo.begin() + 1, PricingInfo.end()) / std::vector<double>(PricingInfo.begin(), PricingInfo.end()-1) - 1;
        // get market price of the 2N+1 days
        map<string, double> market_price=etf.getPrice(SampleVector[i].getPricingInfo());
        Vector market_price_vec = map_values_to_vec(market_price);
        // get market return of the 2N days
        Vector ETFReturn = std::vector<double>(market_price_vec.begin() + 1, market_price_vec.end()) / std::vector<double>(market_price_vec.begin(), market_price_vec.end()-1) - 1;
        // get abnormal return
        AR = ReturnVector - ETFReturn;
        AAR = i*AAR/(i+1) + AR/(i+1);
        }
}

void SampledGroup::calculateCAAR()
{
    int size = AAR.size();
    CAAR = AAR;     // initialize result vector
    for (int i = 1; i<size; i++)
    {
        CAAR[i] += CAAR[i-1];
    }
}

CalculationResult SampledGroup::calculation()
{
    calculateAAR();
    calculateCAAR();
    CalculationResult overAll(AAR, CAAR);
    return overAll;
}


