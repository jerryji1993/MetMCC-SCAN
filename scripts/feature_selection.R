####### ::::: feature selection ::::: #######
library(tidyverse)
library(doParallel)
library(caret)
library(randomForest)

cat("* Registering multiple processors...\n")
cl <- makeCluster(4, outfile="") # discard messages inside workers
registerDoParallel(cl)

mcc_ncdb <- read.csv("./data/Merkel_NCDB_1.csv", row.names = 1)
mcc_ncdb <-  mcc_ncdb %>%
  replace_na(list(LYMPH_VASCULAR_INVASION = 9,
                  DX_RX_STARTED_DAYS = 0,
                  DX_SURG_STARTED_DAYS = 0,
                  DX_DEFSURG_STARTED_DAYS = 0,
                  DX_RAD_STARTED_DAYS = 0,
                  DX_SYSTEMIC_STARTED_DAYS = 0,
                  DX_CHEMO_STARTED_DAYS = 0,
                  RX_SUMM_SCOPE_REG_LN_2012 = 9
                  ))

num_vars <- c("AGE","TUMOR_SIZE","REGIONAL_NODES_POSITIVE","REGIONAL_NODES_EXAMINED",
              "CS_SITESPECIFIC_FACTOR_1", "DX_RX_STARTED_DAYS", "DX_SURG_STARTED_DAYS",
              "DX_DEFSURG_STARTED_DAYS","DX_RAD_STARTED_DAYS", "RAD_REGIONAL_DOSE_CGY",
              "RAD_BOOST_DOSE_CGY","RAD_NUM_TREAT_VOL", "RAD_ELAPSED_RX_DAYS",
              "DX_SYSTEMIC_STARTED_DAYS", "DX_CHEMO_STARTED_DAYS")
cat_vars <- setdiff(colnames(mcc_ncdb), num_vars)
mcc_ncdb <- mcc_ncdb %<>%
  mutate_each(funs(factor(.)),cat_vars)
str(mcc_ncdb)



## Setting seeds
message("* Setting seeds...",appendLF=FALSE)
seed <- 123
seeds <- vector(mode = "list", length = (10*10 + 1))
seeds.length <- length(seeds) -1
for(i in 1:seeds.length){
  seeds[[i]] <- sample.int(n=3^11,3^10) # make it 3^10, generally large enough
}
seeds[[length(seeds)]] <- seed


#### Variable selection ####
message("** Variable selection **\n","\r",appendLF=FALSE)
message("* Setting variable selection control...",appendLF=FALSE)
varselControl <- rfeControl(functions = rfFuncs,
                            method = "repeatedcv",
                            saveDetails = TRUE,
                            number = 10,
                            repeats = 10,
                            seeds = seeds,
                            verbose = TRUE,
                            allowParallel = TRUE
)

message("* Recursive feature elimination...",appendLF=FALSE)
varsel <- rfe(x = mcc_ncdb[,setdiff(colnames(mcc_ncdb),c("CS_SITESPECIFIC_FACTOR_3",
                                                         "REGIONAL_NODES_POSITIVE",
                                                         "REGIONAL_NODES_EXAMINED"
                                                         ))],
              y = mcc_ncdb$CS_SITESPECIFIC_FACTOR_3,
              sizes = c(1:ncol(mcc_ncdb)),
              rfeControl = varselControl,
              na.action = na.roughfix,
              verbose = TRUE)

stopCluster(cl)
