library(waveslim)
library(ggplot2)
library(quantmod)
setwd("C:/destination/to/save/csv")


# get stock data from yahoo-finance
symbol <- "SPY"  
series <- new.env()
getSymbols.yahoo(symbol, env = series, from="2009-01-01", to=Sys.Date() + 1, auto.assign=T) 
stock <- series[[symbol]]
stock <- stock[, "SPY.Close"]

# log returns
returns <- data.frame(na.omit(diff(log(stock))))

# level 3 MODWT-MRA using least-asymmetric daubechies wavelet (symlet) of length 8
MRA <- mra(returns, "la8", method = 'modwt', 3, boundary = "reflective") # reflective boundary to prevent edge effects

D1 <- MRA[["D1"]]
D2 <- MRA[["D2"]]
D3 <- MRA[["D3"]]
S3 <- MRA[["S3"]]

# remove reflected portion of signal
D1 <- head(data.frame(D1), -nrow(returns))
D2 <- head(data.frame(D2), -nrow(returns))
D3 <- head(data.frame(D3), -nrow(returns))
S3 <- head(data.frame(S3), -nrow(returns))

df <- data.frame(D1,D2,D3,S3)

write.csv(df, file = 'modwt_mra.csv')
