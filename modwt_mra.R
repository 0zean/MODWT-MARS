library(waveslim)
library(ggplot2)
library(quantmod)

# path to were you want CSV with MRA results to be saved
setwd("C:/destination/to/save/csv")

# get stock data from yahoo-finance
symbol <- "SPY"
series <- new.env()

getSymbols.yahoo(symbol,
                 env = series,
                 from = "2009-01-01",
                 to = Sys.Date() + 1,
                 auto.assign = TRUE)

stock <- series[[symbol]]
stock <- stock[, "SPY.Close"]

# log returns
returns <- data.frame(na.omit(diff(log(stock))))
returns <- as.ts(returns)


# level 3 MODWT-MRA using least-asymmetric daubechies wavelet of length 8
mra <- mra(returns,
           wf = "la8",
           method = "modwt",
           J = 3,
           boundary = "reflection")


d1 <- mra[["D1"]]
d2 <- mra[["D2"]]
d3 <- mra[["D3"]]
d3 <- mra[["S3"]]

df <- data.frame(d1, d2, d3, d3)

write.csv(df, file = "modwt_mra.csv")
