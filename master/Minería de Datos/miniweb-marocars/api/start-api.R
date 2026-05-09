pr <- plumber::plumb("plumber.R")
host <- "0.0.0.0"
port <- as.integer(Sys.getenv("PORT", unset = "10000"))
pr$run(host = host, port = port)
