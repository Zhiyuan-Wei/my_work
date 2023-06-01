#pragma once
#include <iostream>
#include <vector>
#include <map>

using namespace std;

// Overload << operator for map
template <typename K, typename V>
std::ostream& operator<<(std::ostream& os, const std::map<K, V>& m)
{
    for (const auto& p : m)
    {
        os << "[" << p.first << ": " << p.second << "]" << std::endl;
    }
    return os;
}

template <typename K, typename V>
vector<V> map_values_to_vec(map<K, V> m){
    vector<V> values;
    for (const auto& kvp : m) values.push_back(kvp.second);
    return values;
}

template <typename K, typename V>
map<K, V> mergeMaps(const map<K, V>& stockMap, const map<K, V>& marketMap) {
    map<K, V> mergedMap;
    for (const auto& kvp : stockMap) {
        K key = kvp.first;
        V value = marketMap.at(key);
        mergedMap[key] = value;
    }
    return mergedMap;
}

template <typename K, typename V>
vector<K> map_keys_to_vec(map<K, V> m){
    vector<K> sorted_keys;
    for (const auto& kvp : m) {
        sorted_keys.push_back(kvp.first);
    }
    sort(sorted_keys.begin(), sorted_keys.end());
    return sorted_keys;
}

template <typename V>
int find_index(const vector<V>& vec, const V& str) {
    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] == str) {
            return i;
        }
    }
    return -1;
}
