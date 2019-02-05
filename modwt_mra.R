library(waveslim)
library(ggplot2)
library(quantmod)
setwd("PATH YOU WANT CSV TO SAVE TO")


symbol <- "SPY"  
series <- new.env()
getSymbols.yahoo(symbol, env = series, from="2009-01-01", to=Sys.Date() + 1, auto.assign=T) 
stock <- series[[symbol]]
stock <- stock[, "SPY.Close"]

returns <- na.omit(diff(log(stock)))

MRA <- mra(returns, "la8", method = 'modwt', 3, boundary = "periodic")

D1 <- MRA[["D1"]]
D2 <- MRA[["D2"]]
D3 <- MRA[["D3"]]
S3 <- MRA[["S3"]]

df <- data.frame(D1,D2,D3,S3)

write.csv(df, file = 'modwt_mra.csv')
