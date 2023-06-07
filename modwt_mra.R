library(waveslim)
library(ggplot2)
library(alphavantager)

# path to were you want CSV with MRA results to be saved
setwd(getwd())

# get stock data through Alpha Vantage
key <- readLines("key.txt") # read api key from text file
av_api_key(key)

data <- av_get("SPY",
               av_fun = "TIME_SERIES_DAILY_ADJUSTED",
               outputsize = "full")

data <- data[data$timestamp >= "2009-01-02",]
stock <- data$close


# log returns
returns <- data.frame(na.omit(diff(log(stock))))
returns <- as.ts(returns)


# level 3 MODWT-MRA using least-asymmetric Daubechies wavelet of length 8
mra <- mra(returns,
           wf = "la8",
           method = "modwt",
           J = 3,
           boundary = "reflection")


d1 <- mra[["D1"]]
d2 <- mra[["D2"]]
d3 <- mra[["D3"]]
s3 <- mra[["S3"]]

df <- data.frame(d1, d2, d3, s3)

write.csv(df, file = "modwt_mra.csv", row.names = FALSE)
