#include "Bootstrap.h"
#include "SampledGroup.h"
#include <cmath>

vector<string> random_sample_group(const vector<string>& Group)
{
    vector<string> group_sample(80);
    vector<string> shuffled = Group;
    default_random_engine rng{random_device{}()};
    shuffle(shuffled.begin(), shuffled.end(), rng);
    copy_n(shuffled.begin(), 80, group_sample.begin());
    return group_sample;
}

int Bootstrap::aggregate_N_iter(vector<SampledGroup*> sg_vec, vector<vector<string>>& ThreeGroups) {
    calculated_results.clear();
    calculated_results.resize(n_group);
    for (int g = 0; g < n_group; ++g) {
        // initialize holders for iteration computation
        Vector AAAR, AAR2, AAR_STD, ACAAR, CAAR2, CAAR_STD;
        AAAR.assign(2*n_days, 0.0);
        AAR2.assign(2*n_days, 0.0);
        AAR_STD.assign(2*n_days, 0.0);
        ACAAR.assign(2*n_days, 0.0);
        CAAR2.assign(2*n_days, 0.0);
        CAAR_STD.assign(2*n_days, 0.0);
        
        for(int n = 0; n < n_iter; ++n) {
            vector<string> symbols = random_sample_group(ThreeGroups[g]);
            SampledGroup sample(symbols, *sg_vec[g], etf, n_days);
            CalculationResult cr = sample.calculation();
            
            AAAR = (double(n) * AAAR + cr.AAR) / (double(n) + 1.0);
            AAR2 = (double(n) * AAR2 + (cr.AAR * cr.AAR)) / (double(n) + 1.0);
            ACAAR = (double(n) * ACAAR + cr.CAAR) / (double(n) + 1.0);
            CAAR2 = (double(n) * CAAR2 + (cr.CAAR * cr.CAAR)) / (double(n) + 1.0);
        }
        
        AAR_STD = vec_power((AAR2 - AAAR * AAAR), 0.5) / sqrt(n_iter - 1.0);
        CAAR_STD = vec_power((CAAR2 - ACAAR * ACAAR), 0.5) / sqrt(n_iter - 1.0);
        
        //average struct 
        CalculationResult avg_cr(AAAR, ACAAR);
        CalculationResult std_cr(AAR_STD, CAAR_STD);
        
        calculated_results[g].push_back(avg_cr);
        calculated_results[g].push_back(std_cr);
    }
    // return 0 for completeness
    return 0;
}
