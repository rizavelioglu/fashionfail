**_Table:_** The class distribution in Fashionpedia-train and FashionFail-train.

| class                 | Fashionpedia# | Fashionpedia% | FashionFail# | FashionFail# |
|-----------------------|:-------------:|:-------------:|:------------:|--------------|
| shoe                  |     46374     |    30.31\%    |     441      | 32.81\%      |
| dress                 |     18739     |    12.25\%    |      29      | 2.16\%       |
| top, t-shirt, sweats. |     16548     |    10.82\%    |      71      | 5.28\%       |
| pants                 |     12414     |    8.11\%     |      53      | 3.94\%       |
| jacket                |     7833      |    5.12\%     |      45      | 3.35\%       |
| bag, wallet           |     7217      |    4.72\%     |     146      | 10.86\%      |
| shirt, blouse         |     6161      |    4.03\%     |      57      | 4.24\%       |
| skirt                 |     5046      |    3.30\%     |      26      | 1.93\%       |
| glasses               |     4855      |    3.17\%     |      62      | 4.61\%       |
| tights, stockings     |     4326      |    2.83\%     |      39      | 2.90\%       |
| headband, head c.     |     3470      |    2.27\%     |      45      | 3.35\%       |
| watch                 |     3389      |    2.22\%     |      99      | 7.37\%       |
| coat                  |     3124      |    2.04\%     |      12      | 0.89\%       |
| shorts                |     2756      |    1.80\%     |      52      | 3.87\%       |
| sock                  |     2582      |    1.69\%     |      22      | 1.64\%       |
| hat                   |     2518      |    1.65\%     |      88      | 6.55\%       |
| glove                 |     1385      |    0.91\%     |      12      | 0.89\%       |
| scarf                 |     1374      |    0.90\%     |      2       | 0.15\%       |
| cardigan              |     1107      |    0.72\%     |      1       | 0.07\%       |
| jumpsuit              |      922      |    0.60\%     |      16      | 1.19\%       |
| vest                  |      719      |    0.47\%     |      23      | 1.71\%       |
| umbrella              |      135      |    0.09\%     |      3       | 0.22\%       |
| **TOTAL**             |  **152994**   |   **100\%**   |   **1344**   | **100\%**    |


### Comparison with Fashionpedia

Fashionpedia(FP) has the following splits (with number of images);\
*Train* (45,623), *Validation* (1,158), and *Test* (2,044). \
FashionFail(FF) has the following splits (with number of images);\
*Train* (1,344), *Validation* (150), and *Test* (1,001).

The following table shows the number of samples in each split for each class. For example, *FF-train*
contains *57* samples of `shirt, blouse` class. As FF has exactly 1 annotation per image, this means
there are *57* images (out of *1344*) of `shirt, blouse` class inside *FF-train*. On the contrary, FP
has many annotations per image — *FP-val* has a total of *4481* annotations for *1158* images.

|    | Class / Split      | FF-train | FF-val | FF-test | FP-train⬇ | FP-val |
|----|--------------------|----------|--------|---------|-----------|--------|
| 23 | shoe               | 441      | 49     | 328     | 46374     | 1566   |
| 10 | dress              | 29       | 3      | 22      | 18739     | 508    |
| 1  | top,t-shirt,sweats | 71       | 8      | 53      | 16548     | 477    |
| 6  | pants              | 53       | 6      | 40      | 12414     | 314    |
| 4  | jacket             | 45       | 5      | 34      | 7833      | 183    |
| 24 | bag, wallet        | 146      | 16     | 109     | 7217      | 214    |
| 0  | shirt, blouse      | 57       | 6      | 42      | 6161      | 102    |
| 8  | skirt              | 26       | 3      | 20      | 5046      | 162    |
| 13 | glasses            | 62       | 7      | 46      | 4855      | 130    |
| 21 | tights, stockings  | 39       | 4      | 29      | 4326      | 122    |
| 15 | headband,head c.   | 45       | 5      | 33      | 3470      | 109    |
| 18 | watch              | 99       | 11     | 73      | 3389      | 84     |
| 9  | coat               | 12       | 1      | 8       | 3124      | 104    |
| 7  | shorts             | 52       | 6      | 39      | 2756      | 106    |
| 22 | sock               | 22       | 3      | 17      | 2582      | 87     |
| 14 | hat                | 88       | 10     | 65      | 2518      | 74     |
| 17 | glove              | 12       | 2      | 10      | 1385      | 31     |
| 25 | scarf              | 2        | 0      | 1       | 1374      | 48     |
| 3  | cardigan           | 1        | 0      | 1       | 1107      | 12     |
| 11 | jumpsuit           | 16       | 2      | 12      | 922       | 21     |
| 5  | vest               | 23       | 3      | 18      | 719       | 22     |
| 26 | umbrella           | 3        | 0      | 1       | 135       | 5      |
|    | **TOTAL**          | 1344     | 150    | 1001    | 152994    | 4481   |
