# TWCS Dataset Analysis

Input: `dataset\twcs\twcs.csv`

## Dataset Health

| Metric | Value |
|---|---:|
| Rows | 2811774 |
| Unique tweets | 2811774 |
| Duplicate rows | 0 |
| Duplicate IDs | 0 |
| Unique authors | 702777 |
| Customer tweets | 1537843 |
| Brand tweets | 1273931 |
| Empty messages | 0 |
| Invalid dates | 0 |
| Unique references | 2760421 |
| Broken references | 226103 |
| Date range | 2008-05-08 20:13:59+00:00 to 2017-12-03 23:14:01+00:00 |

## Response Analysis

| Direction | Count |
|---|---:|
| customer_to_brand | 1196963 |
| brand_to_customer | 1266666 |
| customer_to_customer | 235066 |
| brand_to_brand | 3502 |

| Response-time statistic | Seconds | Minutes |
|---|---:|---:|
| count | 1261888 | 21031.47 |
| median | 1271.0 | 21.18 |
| p25 | 372.0 | 6.2 |
| p75 | 6438.0 | 107.3 |
| p90 | 31904.0 | 531.73 |
| p95 | 70998.95 | 1183.32 |
| max | 156584810.0 | 2609746.83 |

## Conversation Statistics

Single-brand conversations: 794946
Multi-brand conversations: 3066

| Statistic | Turns |
|---|---:|
| count | 798012 |
| median | 2.0 |
| p25 | 2.0 |
| p75 | 4.0 |
| p90 | 6.0 |
| p95 | 8.0 |
| max | 1390 |

| Bucket | Count |
|---|---:|
| 1 | 0 |
| 2 | 435270 |
| 3 | 111969 |
| 4 | 104816 |
| 5-10 | 125657 |
| 10+ | 20300 |

## Customer Message Quality

| Metric | Count |
|---|---:|
| 10_25 | 60014 |
| 26_50 | 186297 |
| 51_100 | 443983 |
| emojis | 277142 |
| hashtags | 223265 |
| low_information | 3 |
| mentions | 1957438 |
| over_100 | 846790 |
| questions | 397816 |
| under_10 | 759 |
| urls | 220636 |

## Candidate Brands

Ranking is for manual inspection; no arbitrary score selects a target brand.

| Rank | Brand | Customer messages | Responses | Conversations | Customers | Agent tweet share |
|---:|---|---:|---:|---:|---:|---:|
| 1 | AmazonHelp | 167149 | 168814 | 82534 | 65898 | 13.33% |
| 2 | AppleSupport | 105156 | 106646 | 80702 | 68758 | 8.39% |
| 3 | Uber_Support | 59002 | 56160 | 41923 | 35975 | 4.42% |
| 4 | SpotifyCares | 39690 | 43092 | 28280 | 25055 | 3.40% |
| 5 | AmericanAir | 39186 | 36531 | 26385 | 20187 | 2.89% |
| 6 | Delta | 33816 | 42114 | 26166 | 19396 | 3.32% |
| 7 | TMobileHelp | 30446 | 34215 | 22789 | 16745 | 2.69% |
| 8 | VirginTrains | 29932 | 27416 | 14850 | 11962 | 2.18% |
| 9 | SouthwestAir | 29256 | 28828 | 21636 | 18481 | 2.27% |
| 10 | comcastcares | 27811 | 32921 | 24061 | 18756 | 2.59% |
| 11 | Ask_Spectrum | 24593 | 25617 | 18530 | 15479 | 2.03% |
| 12 | British_Airways | 22944 | 29290 | 16450 | 11858 | 2.30% |
| 13 | hulu_support | 22609 | 21681 | 14955 | 12801 | 1.72% |
| 14 | ATVIAssist | 22304 | 17514 | 11096 | 13711 | 1.39% |
| 15 | GWRHelp | 21541 | 19237 | 10728 | 7289 | 1.52% |
| 16 | Tesco | 21464 | 38468 | 16721 | 11752 | 3.03% |
| 17 | AskPlayStation | 20569 | 18675 | 12533 | 11887 | 1.50% |
| 18 | XboxSupport | 20533 | 23235 | 13454 | 11145 | 1.93% |
| 19 | ChipotleTweets | 19578 | 18599 | 14392 | 13431 | 1.47% |
| 20 | VerizonSupport | 19410 | 17805 | 8446 | 7014 | 1.41% |

Top 5 brands contain 32.42% of agent tweets.
Top 10 brands contain 45.48% of agent tweets.

## Risks

- Broken references are computed after all tweet IDs are collected, so CSV order cannot create false positives.
- The final observed direction is not proof that an issue was resolved.
- Exclude or separately handle multi-brand conversations for target-brand training and evaluation.
