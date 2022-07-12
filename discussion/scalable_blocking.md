# Scalable Blocking Using H3 Geospatial Index

I share a scalable blocking technique using a geospatial index.
In many public notebooks, the dataset is partitioned by country to limit the search space of the k-nearest neighbor.
The problem with this simple division is that the sample size is biased by country.
This reduces scalability as the number of data points increases.
Therefore, I was investigating techniques to split the earth segment independent of country.
A promising tool I found is [uber/H3]. uber/H3 maps latitude and longitude to individual location IDs.
Using this, the earth can be more flexibly divided into a hexagonal grid of various sizes (see Table 1).

The search and query spaces are shown in Figure 1; if a location belongs to a H3 grid, I choose the search space within one-distance neighborhood of the query space. That ensures the area of search space is approximately bigger than that of a H3 grid (which is chosen to be bigger than expected size of POIs).

Since I already know that more than 99% of the points are within a radius of 26 km (see Tab. 2), a H3 resolution of 4 or less is sufficient to create a good blocker.
In addition, the study on execution time (Tab. 3) determined that resolution 1 is optimal for the current competition data.

[uber/H3]: https://github.com/uber/h3

## Entire Solution Code

I shared entire [solution code].
(Unfortunately, the inference notebook is not scored due to error.)

[solution code]: https://github.com/bilzard/kaggle-4sq-location-matching


<center><b>Tab.1: Resolution & Hex Size</b></center>

| H3 Resolution | Average Hexagon Area (km2) | Average Hexagon Edge Length (km) | Number of unique indexes |
| :-----------: | -------------------------- | -------------------------------- | ------------------------ |
|       0       | 4,250,546.8477000          | 1,107.712591000                  | 122                      |
|       1       | 607,220.9782429            | 418.676005500                    | 842                      |
|       2       | 86,745.8540347             | 158.244655800                    | 5,882                    |
|       3       | 12,392.2648621             | 59.810857940                     | 41,162                   |
|       4       | 1,770.3235517              | 22.606379400                     | 288,122                  |
|       5       | 252.9033645                | 8.544408276                      | 2,016,842                |
|       6       | 36.1290521                 | 3.229482772                      | 14,117,882               |
|       7       | 5.1612932                  | 1.220629759                      | 98,825,162               |
|       8       | 0.7373276                  | 0.461354684                      | 691,776,122              |
|       9       | 0.1053325                  | 0.174375668                      | 4,842,432,842            |
|      10       | 0.0150475                  | 0.065907807                      | 33,897,029,882           |
|      11       | 0.0021496                  | 0.024910561                      | 237,279,209,162          |
|      12       | 0.0003071                  | 0.009415526                      | 1,660,954,464,122        |
|      13       | 0.0000439                  | 0.003559893                      | 11,626,681,248,842       |
|      14       | 0.0000063                  | 0.001348575                      | 81,386,768,741,882       |
|      15       | 0.0000009                  | 0.000509713                      | 569,707,381,193,162      |

source: https://h3geo.org/docs/core-library/restable/

<center><b>Tab.2: Distribution of POI Size</b></center>

| size_km  | cum_dist |
| -------- | -------- |
| 0.000    | 0.5000   |
| 0.062    | 0.7500   |
| 0.410    | 0.8750   |
| 1.436    | 0.9375   |
| 4.360    | 0.9688   |
| 11.230   | 0.9844   |
| 26.413   | 0.9922   |
| 72.365   | 0.9961   |
| 269.149  | 0.9980   |
| 1024.761 | 0.9990   |

source: https://www.kaggle.com/code/tatamikenn/4sq-eda-on-poi/notebook

<center><b>Tab3.: Execution Time v.s. H3 Resolution</b></center>

| H3 resolution | execution time (sec) |
| ------------- | -------------------- |
| 5             | 7,200                |
| 4             | 3,000                |
| 3             | 1,140                |
| 2             | 347                  |
| 1             | 213                  |
| 0             | 275                  |


<a href="https://ibb.co/5cZvnXB"><img src="https://i.ibb.co/ys25YmW/h3-grids.png" alt="h3-grids" border="0"></a>

<center><b>Fig.1: Query Space & Search Space Visualized</b></center>


[uber/H3]: https://github.com/uber/h3
