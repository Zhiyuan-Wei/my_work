%% get and combine live and dead RI, and get gross return

% get US live fund data
clear ID Ind opts2 
opts2 = spreadsheetImportOptions('Numvariables',465);
opts2.Sheet = 'total return index';
opts2.VariableTypes(1:2) = {'string'};
opts2.VariableTypes(3:end) = {'double'};
opts2.DataRange = 'A2:QW7409';
USliveRI = readtable('D:\Desktop\school files\University of Birmingham\course\university course\senior semester 2\economic extended essay\MutualFundData\MutualFundData2\US Data\USLiveEquityFunds.xlsx',opts2);
USliveRI(:,3:13) = [];
USliveRI(:,277:end) = [];

% get US dead fund data
clear ID Ind opts2
opts2 = spreadsheetImportOptions('Numvariables',627);
opts2.Sheet = 'total return index';
opts2.VariableTypes(1:2) = {'string'};
opts2.VariableTypes(3:end) = {'double'};
opts2.DataRange = 'A2:XC1617';
USDeadRI = readtable('D:\Desktop\school files\University of Birmingham\course\university course\senior semester 2\economic extended essay\MutualFundData\MutualFundData2\US Data\USDeadEquityFunds.xlsx', opts2);
USDeadRI(:,3:175) = [];
USDeadRI(:,277:end) = [];

nRowCol = size(USDeadRI);
for ind = 1:nRowCol(1)
    md = mode(USDeadRI{ind,3:end},'all');
    a = USDeadRI{ind,3:end} == md;
    a = logical([0 0 a]);
    USDeadRI{ind,a} = NaN;
end

% change variable name to combine
allVars = 1:width(USDeadRI);
newNames = append("Reading",string(allVars));
USliveRI = renamevars(USliveRI,allVars,newNames);
USDeadRI = renamevars(USDeadRI,allVars,newNames);

% combine live and dead funds and calculate gross return
USRI = [USliveRI; USDeadRI];
USRIName = USRI(:,1);  % get fund name
USId = USRI(:,2);  % get fund id
USRI(:,1) = [];
nRowCol = size(USRI);
USRetGrs = NaN(nRowCol(1),(nRowCol(2)-1));
for ind = 1:nRowCol(1)
    row = table2array(USRI(ind,2:end));
    notnan = ~isnan(row);
    noNaN  = row(notnan);
    grt = NaN(1,length(noNaN));
    for in = 2:length(noNaN)
        grt(in) = (noNaN(in)/noNaN(in-1))-1;
    end   
    row(notnan) = grt;
    zero = row == 0;
    row(zero) = NaN;
    USRetGrs(ind,2:end) = row(2:end);
end

%% get and combine live and dead TER

% get US live fund data
clear ID Ind opts2 
opts2 = spreadsheetImportOptions('Numvariables',465);
opts2.Sheet = 'total expense ratio';
opts2.VariableTypes(1:2) = {'string'};
opts2.VariableTypes(3:end) = {'double'};
opts2.DataRange = 'A2:QW7409';
USliveTER = readtable('D:\Desktop\school files\University of Birmingham\course\university course\senior semester 2\economic extended essay\MutualFundData\MutualFundData2\US Data\USLiveEquityFunds.xlsx',opts2);
USliveTER(:,2:13) = [];
USliveTER(:,276:end) = [];

% get US dead fund data
clear ID Ind opts2 
opts2 = spreadsheetImportOptions('Numvariables',627);
opts2.Sheet = 'total expense ratio';
opts2.VariableTypes(1:2) = {'string'};
opts2.VariableTypes(3:end) = {'double'};
opts2.DataRange = 'A2:XC1617';
USDeadTER = readtable('D:\Desktop\school files\University of Birmingham\course\university course\senior semester 2\economic extended essay\MutualFundData\MutualFundData2\US Data\USDeadEquityFunds.xlsx', opts2);
USDeadTER(:,2:175) = [];
USDeadTER(:,276:end) = [];


nRowCol = size(USDeadTER);
for ind = 1:nRowCol(1)
    md = mode(USDeadTER{ind,2:end},'all');
    a = USDeadTER{ind,2:end} == md;
    a = logical([0 a]);
    USDeadTER{ind,a} = NaN;
end

allVars = 1:width(USDeadTER);
newNames = append("Reading",string(allVars));
USliveTER = renamevars(USliveTER,allVars,newNames);
USDeadTER = renamevars(USDeadTER,allVars,newNames);

% combine the two dataset
USTER = [USliveTER; USDeadTER];
USTERTrim = USTER;
USTERTrim(:,2) = []; % USTERTrim is USTER without data for Dec 1983

%% get and combine live and dead TNA

% get US live fund data
clear ID Ind opts2 
opts2 = spreadsheetImportOptions('Numvariables',465);
opts2.Sheet = 'total net asset';
opts2.VariableTypes(1:2) = {'string'};
opts2.VariableTypes(3:end) = {'double'};
opts2.DataRange = 'A2:QW7409';
USliveTNA = readtable('D:\Desktop\school files\University of Birmingham\course\university course\senior semester 2\economic extended essay\MutualFundData\MutualFundData2\US Data\USLiveEquityFunds.xlsx',opts2);
USliveTNA(:,2:13) = [];
USliveTNA(:,276:end) = [];

% get US dead fund data
clear ID Ind opts2
opts2 = spreadsheetImportOptions('Numvariables',627);
opts2.Sheet = 'total net asset';
opts2.VariableTypes(1:2) = {'string'};
opts2.VariableTypes(3:end) = {'double'};
opts2.DataRange = 'A2:XC1617';
USDeadTNA = readtable('D:\Desktop\school files\University of Birmingham\course\university course\senior semester 2\economic extended essay\MutualFundData\MutualFundData2\US Data\USDeadEquityFunds.xlsx', opts2);
USDeadTNA(:,2:175) = [];
USDeadTNA(:,276:end) = [];

nRowCol = size(USDeadTNA);
for ind = 1:nRowCol(1)
    md = mode(USDeadTNA{ind,2:end},'all');
    a = USDeadTNA{ind,2:end} == md;
    a = logical([0 a]);
    USDeadTNA{ind,a} = NaN;
end

allVars = 1:width(USDeadTNA);
newNames = append("Reading",string(allVars));
USliveTNA = renamevars(USliveTNA,allVars,newNames);
USDeadTNA = renamevars(USDeadTNA,allVars,newNames);

% combine the two dataset
USTNA = [USliveTNA; USDeadTNA];
USTNATrim = USTNA;
USTNATrim(:,2) = []; % USTNATrim is USTNA without data for Dec 1983

%% read factor data
opts = spreadsheetImportOptions('Numvariables',7);
opts.Sheet = '5factor';
opts.VariableTypes = {'string'  'double' 'double' 'double' 'double' 'double' 'double'};
opts.DataRange = 'A251:G523';
US5Factor = readtable('D:\Desktop\school files\University of Birmingham\course\university course\senior semester 2\economic extended essay\MutualFundData\MutualFundData2\US Data\USFactors.xlsx', opts);
US3Factor = US5Factor;
US3Factor(:,5:6) = [];

%% read factor data with information variable

% conditional model based on 3 factor CAPM
clear ID Ind opts
opts = spreadsheetImportOptions('Numvariables',9);
opts.Sheet = '3fcon';
opts.VariableTypes(1) = {'string'};
opts.VariableTypes(2:end) = {'double'};
opts.DataRange = 'A1:I273';
US3Factor_con = readtable('D:\Desktop\school files\University of Birmingham\course\university course\senior semester 2\economic extended essay\MutualFundData\MutualFundData2\US Data\conFactors.xlsx', opts);

% conditional model based on 5 factor CAPM
clear ID Ind opts
opts = spreadsheetImportOptions('Numvariables',11);
opts.Sheet = '5fcon';
opts.VariableTypes(1) = {'string'};
opts.VariableTypes(2:end) = {'double'};
opts.DataRange = 'A1:K273';
US5Factor_con = readtable('D:\Desktop\school files\University of Birmingham\course\university course\senior semester 2\economic extended essay\MutualFundData\MutualFundData2\US Data\conFactors.xlsx', opts);

%% get net return

USTER_temp = USTER(:,2:end);
USRI_temp = USRI(:,2:end);
nRowCol = size(USRI);
USRetNet = NaN(nRowCol(1),nRowCol(2)-1);
row1 = NaN(1,(nRowCol(2)-1));
row2 = NaN(1,(nRowCol(2)-1));
row4 = NaN(1,(nRowCol(2)-1));
j=1;
for ind = 1:nRowCol(1)
          notnan          = ~isnan(table2array(USRI(ind,2:end))); %274
          row1(notnan)    = table2array(USRI_temp(ind,notnan)); %274
          rows            = row1(notnan); %274
          row2(notnan)    = [NaN rows(2:end)]; %274
          row4(notnan)    = [NaN rows(1:(end-1))]; %274
          ret(j,notnan)   = row2(notnan)./row4(notnan)-1; %274 Nan
          row3            = table2array(USTER_temp(ind,notnan)); %273
          row3nonan = ~isnan(row3);

          mrow3 = mean(row3(row3nonan),'all');
          for ind2 = 1:length(row3) 
             
               if isnan(row3(ind2))
                  row3(ind2) = mrow3;
               end
            
          end
          USRetNet(j,notnan) = ret(j,notnan) - row3./1200;
          
   j = j+1;
end

%% remove funds with no data and exclude passively managed funds
nColRowUSRI = size(USRI);  %9024*274
array_USRetGrs = USRetGrs;
USRetGrs = array2table(USRetGrs);
array_USRetNet = USRetNet;
USRetNet = array2table(USRetNet);
array_USTNATrim = table2array(USTNATrim(:,2:end));
array_USTNA = table2array(USTNA(:,2:end));
array_USRIName = table2array(USRIName);

% funds with any of these keywords are regarded as enhanced index fund or target-date fund
Enhanced_index_fund = ["INDEX","INDX","IDX","S&P","SCHWAB 1000","NASDAQ-100","DOW","JONES","ETF","ISHARE",...
    "PROFOUND","RUSSELL","PROSHARE","POWERSHARE","VIPER","SPIDER","SPDR","WILSHARE","ETN","EXCHANGE TRADED",...
    "EXCHANGE-TRADED","HEDGE","MANAGE","ENHANCE","PLUS","1970","1975","1980","1985","1990","1995","2000",...
    "2005","2010","2015","2020","2025","2030","2035","2040","2045","2050","2055","2060","TARGET","LIFESTYLE"];

 % funds with any of these keywords in leveraged_index_fund_in and none of
 % the keyywords in leveraged_index_fund_ex are regarded as leveraged index fund
leveraged_index_fund_in = ["INVERSE","SHORT","ULTRA","1.25X","1.5X","2X","2.5X","3X","4X","5X","6X","7X","8X","9X","0X"];
leveraged_index_fund_ex = ["SHORT TERM","SHORT TM","SHORT BOND","SHORT BND","SHORT BD","LG SHORT","LONG/SHORT",...
    "LONG-SHORT","LONG SHORT"];

% conduct fund filter to filter funds with no data and passively managed funds
array_USRetGrs_filter = array_USRetGrs(sum(array_USRetNet > -100,2) > 0 & sum(array_USTNA > 0,2) > 0 & ~contains(array_USRIName,Enhanced_index_fund) & ~(contains(array_USRIName,leveraged_index_fund_in) & ~contains(array_USRIName,leveraged_index_fund_ex)),:);
array_USRetNet_filter = array_USRetNet(sum(array_USRetNet > -100,2) > 0 & sum(array_USTNA > 0,2) > 0 & ~contains(array_USRIName,Enhanced_index_fund) & ~(contains(array_USRIName,leveraged_index_fund_in) & ~contains(array_USRIName,leveraged_index_fund_ex)),:);
array_USTNATrim_filter = array_USTNATrim(sum(array_USRetNet > -100,2) > 0 & sum(array_USTNA > 0,2) > 0 & ~contains(array_USRIName,Enhanced_index_fund) & ~(contains(array_USRIName,leveraged_index_fund_in) & ~contains(array_USRIName,leveraged_index_fund_ex)),:);
USRIName_filter = USRIName(sum(array_USRetNet > -100,2) > 0 & sum(array_USTNA > 0,2) > 0 & ~contains(array_USRIName,Enhanced_index_fund) & ~(contains(array_USRIName,leveraged_index_fund_in) & ~contains(array_USRIName,leveraged_index_fund_ex)),:);
USId_filter = USId(sum(array_USRetNet > -100,2) > 0 & sum(array_USTNA > 0,2) > 0 & ~contains(array_USRIName,Enhanced_index_fund) & ~(contains(array_USRIName,leveraged_index_fund_in) & ~contains(array_USRIName,leveraged_index_fund_ex)),:);

%% combine the same fund in different classes (using funds' ids)

% get the position of funds which change classes
nColRowarray_USRetNet_filter = size(array_USRetNet_filter);
diff_sig = ones(1,nColRowarray_USRetNet_filter(1)+1);
for i=2:nColRowarray_USRetNet_filter(1)
    char_seq_1 = convertStringsToChars(string(USId_filter.Reading2(i,1)));
    char_seq_2 = convertStringsToChars(string(USId_filter.Reading2(i-1,1)));
    if char_seq_1(1:4) == char_seq_2(1:4)
        diff_sig(1,i) = 0;
    end
end

% get AGTNA
diff_index = find(diff_sig);
USAGTNA = array_USTNATrim_filter;
USAGTNA = array2table(USAGTNA);
diff_index_length = length(diff_index);
for i=1:diff_index_length-1
    temp = array2table(sum(array_USTNATrim_filter(diff_index(i):diff_index(i+1)-1,:),1));
    for j=diff_index(i):diff_index(i+1)-1
        USAGTNA(j,:) = temp;
    end
end

% get combined fund return
weight = array_USTNATrim_filter./table2array(USAGTNA);
temp_RetGrs = weight.*array_USRetGrs_filter(:,2:end);
temp_RetNet = weight.*array_USRetNet_filter(:,2:end);
RetGrs = USRetGrs(1:diff_index_length-1,:);
RetNet = USRetNet(1:diff_index_length-1,:);
for i=1:diff_index_length-1
    temp = array2table(sum(temp_RetGrs(diff_index(i):diff_index(i+1)-1,:),1));
    RetGrs(i,2:end) = temp;
end
for i=1:diff_index_length-1
    temp = array2table(sum(temp_RetNet(diff_index(i):diff_index(i+1)-1,:),1));
    RetNet(i,2:end) = temp;
end

% get unique AGTNA
AGTNA_unique = USAGTNA(1:diff_index_length-1,:);
for i=1:diff_index_length-1
    AGTNA_unique(i,:) = USAGTNA(diff_index(i),:);
end

%% set TNA threshold get get three groups of funds (5 million, 250 million and 1 billion)

array_AGTNA_unique = double(table2array(AGTNA_unique));
nRowCol = size(RetGrs);

temp = array_AGTNA_unique >= 5;
temp2 = double(temp);
temp = zeros(nRowCol(1),nRowCol(2));
temp(:,2:end) = temp2;
temp(temp==0)=NaN;
temp = fillmissing(temp,'previous',2);
RetGrs_above5 = table2array(RetGrs).*temp;
RetGrs_above5 = RetGrs_above5(array_AGTNA_unique(:,end) >= 5,:);
RetGrs_above5 = array2table(RetGrs_above5);
RetNet_above5 = table2array(RetNet).*temp;
RetNet_above5 = RetNet_above5(array_AGTNA_unique(:,end) >= 5,:);
RetNet_above5 = array2table(RetNet_above5);

temp = array_AGTNA_unique >= 250;
temp2 = double(temp);
temp = zeros(nRowCol(1),nRowCol(2));
temp(:,2:end) = temp2;
temp(temp==0)=NaN;
temp = fillmissing(temp,'previous',2);
RetGrs_above250 = table2array(RetGrs).*temp;
RetGrs_above250 = RetGrs_above250(array_AGTNA_unique(:,end) >= 250,:);
RetGrs_above250 = array2table(RetGrs_above250);
RetNet_above250 = table2array(RetNet).*temp;
RetNet_above250 = RetNet_above250(array_AGTNA_unique(:,end) >= 250,:);
RetNet_above250 = array2table(RetNet_above250);

temp = array_AGTNA_unique >= 1000;
temp2 = double(temp);
temp = zeros(nRowCol(1),nRowCol(2));
temp(:,2:end) = temp2;
temp(temp==0)=NaN;
temp = fillmissing(temp,'previous',2);
RetGrs_above1000 = table2array(RetGrs).*temp;
RetGrs_above1000 = RetGrs_above1000(array_AGTNA_unique(:,end) >= 1000,:);
RetGrs_above1000 = array2table(RetGrs_above1000);
RetNet_above1000 = table2array(RetNet).*temp;
RetNet_above1000 = RetNet_above1000(array_AGTNA_unique(:,end) >= 1000,:);
RetNet_above1000 = array2table(RetNet_above1000);

%% get percentiles of t(alpha)s unconditional

% get each fund's t-value under each of the 12 sets (gross or net; TNA
% threshold and 3 factor CAPM or 5 factor CAPM
nRowCol = size(RetGrs_above5);
Tstat_gross_above5_3factor = calculatealpha12Aug2021(RetGrs_above5,US3Factor);
Tstat_gross_above250_3factor = calculatealpha12Aug2021(RetGrs_above250,US3Factor);
Tstat_gross_above1000_3factor = calculatealpha12Aug2021(RetGrs_above1000,US3Factor);
Tstat_gross_above5_5factor = calculatealpha12Aug2021(RetGrs_above5,US5Factor);
Tstat_gross_above250_5factor = calculatealpha12Aug2021(RetGrs_above250,US5Factor);
Tstat_gross_above1000_5factor = calculatealpha12Aug2021(RetGrs_above1000,US5Factor);
Tstat_net_above5_3factor = calculatealpha12Aug2021(RetNet_above5,US3Factor);
Tstat_net_above250_3factor = calculatealpha12Aug2021(RetNet_above250,US3Factor);
Tstat_net_above1000_3factor = calculatealpha12Aug2021(RetNet_above1000,US3Factor);
Tstat_net_above5_5factor = calculatealpha12Aug2021(RetNet_above5,US5Factor);
Tstat_net_above250_5factor = calculatealpha12Aug2021(RetNet_above250,US5Factor);
Tstat_net_above1000_5factor = calculatealpha12Aug2021(RetNet_above1000,US5Factor);

% put the t(alpha)s into a single table
t_table_5factor = zeros(nRowCol(1),12)*nan;
t_table_5factor(1:length(Tstat_gross_above5_3factor),1) = Tstat_gross_above5_3factor;
t_table_5factor(1:length(Tstat_gross_above250_3factor),2) = Tstat_gross_above250_3factor;
t_table_5factor(1:length(Tstat_gross_above1000_3factor),3) = Tstat_gross_above1000_3factor;
t_table_5factor(1:length(Tstat_gross_above5_5factor),4) = Tstat_gross_above5_5factor;
t_table_5factor(1:length(Tstat_gross_above250_5factor),5) = Tstat_gross_above250_5factor;
t_table_5factor(1:length(Tstat_gross_above1000_5factor),6) = Tstat_gross_above1000_5factor;
t_table_5factor(1:length(Tstat_net_above5_3factor),7) = Tstat_net_above5_3factor;
t_table_5factor(1:length(Tstat_net_above250_3factor),8) = Tstat_net_above250_3factor;
t_table_5factor(1:length(Tstat_net_above1000_3factor),9) = Tstat_net_above1000_3factor;
t_table_5factor(1:length(Tstat_net_above5_5factor),10) = Tstat_net_above5_5factor;
t_table_5factor(1:length(Tstat_net_above250_5factor),11) = Tstat_net_above250_5factor;
t_table_5factor(1:length(Tstat_net_above1000_5factor),12) = Tstat_net_above1000_5factor;

% get t-value percentile table
percentile = [1,2,3,4,5,10,20,30,40,50,60,70,80,90,95,96,97,98,99];
for i=1:19
    t_table_percentile_5factor(i,:) = prctile(t_table_5factor,percentile(i));
end

%% get percentiles of t(alpha)s conditional

% get each fund's t-value under each of the 12 sets (gross or net; TNA
% threshold and 3 factor CAPM or 5 factor CAPM
Tstat_gross_above5_3factor_con = calculatealpha12Aug2021(RetGrs_above5,US3Factor_con);
Tstat_gross_above250_3factor_con = calculatealpha12Aug2021(RetGrs_above250,US3Factor_con);
Tstat_gross_above1000_3factor_con = calculatealpha12Aug2021(RetGrs_above1000,US3Factor_con);
Tstat_gross_above5_5factor_con = calculatealpha12Aug2021(RetGrs_above5,US5Factor_con);
Tstat_gross_above250_5factor_con = calculatealpha12Aug2021(RetGrs_above250,US5Factor_con);
Tstat_gross_above1000_5factor_con = calculatealpha12Aug2021(RetGrs_above1000,US5Factor_con);
Tstat_net_above5_3factor_con = calculatealpha12Aug2021(RetNet_above5,US3Factor_con);
Tstat_net_above250_3factor_con = calculatealpha12Aug2021(RetNet_above250,US3Factor_con);
Tstat_net_above1000_3factor_con = calculatealpha12Aug2021(RetNet_above1000,US3Factor_con);
Tstat_net_above5_5factor_con = calculatealpha12Aug2021(RetNet_above5,US5Factor_con);
Tstat_net_above250_5factor_con = calculatealpha12Aug2021(RetNet_above250,US5Factor_con);
Tstat_net_above1000_5factor_con = calculatealpha12Aug2021(RetNet_above1000,US5Factor_con);

% put the t(alpha)s into a single table
t_table_con = zeros(nRowCol(1),12)*nan;
t_table_con(1:length(Tstat_gross_above5_3factor_con),1) = Tstat_gross_above5_3factor_con;
t_table_con(1:length(Tstat_gross_above250_3factor_con),2) = Tstat_gross_above250_3factor_con;
t_table_con(1:length(Tstat_gross_above1000_3factor_con),3) = Tstat_gross_above1000_3factor_con;
t_table_con(1:length(Tstat_gross_above5_5factor_con),4) = Tstat_gross_above5_5factor_con;
t_table_con(1:length(Tstat_gross_above250_5factor_con),5) = Tstat_gross_above250_5factor_con;
t_table_con(1:length(Tstat_gross_above1000_5factor_con),6) = Tstat_gross_above1000_5factor_con;
t_table_con(1:length(Tstat_net_above5_3factor_con),7) = Tstat_net_above5_3factor_con;
t_table_con(1:length(Tstat_net_above250_3factor_con),8) = Tstat_net_above250_3factor_con;
t_table_con(1:length(Tstat_net_above1000_3factor_con),9) = Tstat_net_above1000_3factor_con;
t_table_con(1:length(Tstat_net_above5_5factor_con),10) = Tstat_net_above5_5factor_con;
t_table_con(1:length(Tstat_net_above250_5factor_con),11) = Tstat_net_above250_5factor_con;
t_table_con(1:length(Tstat_net_above1000_5factor_con),12) = Tstat_net_above1000_5factor_con;

% get t-value percentile table
percentile = [1,2,3,4,5,10,20,30,40,50,60,70,80,90,95,96,97,98,99];
for i=1:19
    t_table_percentile_con(i,:) = prctile(t_table_con,percentile(i));
end

%% get mean and standard deviation of t-values (unconditional and conditional respectively)
mean_and_std_table(1,1:12) = mean(t_table_5factor,’omitnan’);
mean_and_std_table(2,1:12) = std(t_table_5factor,’omitnan’);
mean_and_std_table(3,1:12) = mean(t_table_con,’omitnan’);
mean_and_std_table(4,1:12) = std(t_table_con,’omitnan’);

%% plot t(alpha) distributions
% gross_3factor
subplot(2,3,1)
histogram(t_table_5factor(:,1),30)
title('unconditional 5 million threshold')
subplot(2,3,2)
histogram(t_table_5factor(:,2),30)
title('unconditional 250 million threshold')
subplot(2,3,3)
histogram(t_table_5factor(:,3),30)
title('unconditional 1 billion threshold')
subplot(2,3,4)
histogram(t_table_con(:,1),30)
title('conditional 5 million threshold')
subplot(2,3,5)
histogram(t_table_con(:,2),30)
title('conditional 250 million threshold')
subplot(2,3,6)
histogram(t_table_con(:,3),30)
title('conditional 1 billion threshold')

% gross_5factor
subplot(2,3,1)
histogram(t_table_5factor(:,4),30)
title('unconditional 5 million threshold')
subplot(2,3,2)
histogram(t_table_5factor(:,5),30)
title('unconditional 250 million threshold')
subplot(2,3,3)
histogram(t_table_5factor(:,6),30)
title('unconditional 1 billion threshold')
subplot(2,3,4)
histogram(t_table_con(:,4),30)
title('conditional 5 million threshold')
subplot(2,3,5)
histogram(t_table_con(:,5),30)
title('conditional 250 million threshold')
subplot(2,3,6)
histogram(t_table_con(:,6),30)
title('conditional 1 billion threshold')

% net_3factor
subplot(2,3,1)
histogram(t_table_5factor(:,7),30)
title('unconditional 5 million threshold')
subplot(2,3,2)
histogram(t_table_5factor(:,8),30)
title('unconditional 250 million threshold')
subplot(2,3,3)
histogram(t_table_5factor(:,9),30)
title('unconditional 1 billion threshold')
subplot(2,3,4)
histogram(t_table_con(:,7),30)
title('conditional 5 million threshold')
subplot(2,3,5)
histogram(t_table_con(:,8),30)
title('conditional 250 million threshold')
subplot(2,3,6)
histogram(t_table_con(:,9),30)
title('conditional 1 billion threshold')

% net_5factor
subplot(2,3,1)
histogram(t_table_5factor(:,10),30)
title('unconditional 5 million threshold')
subplot(2,3,2)
histogram(t_table_5factor(:,11),30)
title('unconditional 250 million threshold')
subplot(2,3,3)
histogram(t_table_5factor(:,12),30)
title('unconditional 1 billion threshold')
subplot(2,3,4)
histogram(t_table_con(:,10),30)
title('conditional 5 million threshold')
subplot(2,3,5)
histogram(t_table_con(:,11),30)
title('conditional 250 million threshold')
subplot(2,3,6)
histogram(t_table_con(:,12),30)
title('conditional 1 billion threshold')
