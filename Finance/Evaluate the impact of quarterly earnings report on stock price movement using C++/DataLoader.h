#pragma once
#include <stdio.h>
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <locale>
#include <iomanip>
#include <fstream>
#include <map>
#include "curl/curl.h"
//
using namespace std;

static int write_data(void* ptr, int size, int nmemb, FILE* stream)
{
    size_t written;
    written = fwrite(ptr, size, nmemb, stream);
    return written;
}

// define the struct to store the data
struct MemoryStruct {
    char* memory;
    size_t size;
};

// to allocate the memory
// void* to any pointer type
static void* myrealloc(void* ptr, size_t size)
{
    if (ptr)
        return realloc(ptr, size);
    else
        return malloc(size);
}

// callback function to write the data to a memory chunk
static int write_data2(void* ptr, size_t size, size_t nmemb, void* data)
{
    // calculate the real size of the incoming buffer
    size_t realsize = size * nmemb;
    // create a pointer to the MemoryStruct structure
//	struct MemoryStruct* mem = (struct MemoryStruct*)data;
    MemoryStruct* mem = ( MemoryStruct*)data;
    mem->memory = (char*)myrealloc(mem->memory, mem->size + realsize + 1);
    if (mem->memory) {
        memcpy(&(mem->memory[mem->size]), ptr, realsize);
        mem->size += realsize;
        mem->memory[mem->size] = 0;
    }
    return realsize;
}


//static int GetSingleStockPrice(string StartDate, string EndDate, string Symbol, map<string, double> &PriceInfoMap, map<string, double> &CumReturnInfoMap, CURL * handle)
static int GetSingleStockPrice(string StartDate, string EndDate, string Symbol, map<string, double> &PriceInfoMap, CURL * handle)
{ 

    CURLcode result;
    // if everything's all right with the easy handle...
    if (handle)
    {
        string url_common = "https://eodhistoricaldata.com/api/eod/";
        string start_date = StartDate;
        string end_date = EndDate;
        // You must replace this API token with yours
        string api_token = "644c0116a93c40.56261278";

        
        struct MemoryStruct data;
        data.memory = NULL;
        data.size = 0;
        string symbol = Symbol;
        string url_request = url_common + symbol + ".US?" + "from=" + start_date + "&to=" + end_date + "&api_token="
                            + api_token + "&period=d";
        // set the URL
        curl_easy_setopt(handle, CURLOPT_URL, url_request.c_str());
        //adding a user agent
        curl_easy_setopt(handle, CURLOPT_USERAGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:74.0) Gecko/20100101 Firefox/74.0");
        curl_easy_setopt(handle, CURLOPT_SSL_VERIFYPEER, 0);
        curl_easy_setopt(handle, CURLOPT_SSL_VERIFYHOST, 0);
        
        // set the callback function to write the data to a memory chunk
        curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, write_data2);
        curl_easy_setopt(handle, CURLOPT_WRITEDATA, (void*)&data);
        result = curl_easy_perform(handle);
        
        if (result != CURLE_OK)
        {
            // if errors have occurred, tell what is wrong with result
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(result));
            return 1;
        }
        
        
        stringstream sData;
        sData.str(data.memory);
        string sValue, sDate;
        double dValue = 0;
        string line;
//        bool indicator = true;  // denote whether it is first day
//        double ret = 1.0;
//        double InitdValue = 1.0;
        // cout << symbol << endl;
        while (getline(sData, line)) {
            size_t found = line.find('-');
            if (found != std::string::npos)
            { 
                // cout << line << endl;
                sDate = line.substr(0, line.find_first_of(','));
                line.erase(line.find_last_of(','));
                sValue = line.substr(line.find_last_of(',') + 1).substr();
                dValue = strtod(sValue.c_str(), NULL);
                // ret = dValue / InitdValue;
                // if (indicator) { 
                //     ret = 1.0;
                //     InitdValue = dValue;
                //     indicator = false;
                // }
                //cout << sDate << " " << std::fixed << ::setprecision(4) << dValue << endl;
                PriceInfoMap.insert(make_pair(sDate, dValue));
//                CumReturnInfoMap.insert(make_pair(sDate, ret));
            }
        }
        if ((PriceInfoMap.count(StartDate) != 1))  {
            
            cout << "Warning：" << Symbol << " only have " << PriceInfoMap.size() << " days of data." << endl;
            return -2;
        } 
        
        free(data.memory);
        data.size = 0;
        
    }
    else
    {
        fprintf(stderr, "Curl init failed!\n");
        return -1;
    }
    
    return 0;
}
