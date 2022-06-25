# Panel regression of growth and temperature
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

# Libraries
library(ggplot2)
library(tidyr)
library(lfe)
library(dplyr)

# Locations
loc_panel <- "../Data/Panel/"
loc_out <- "../Data/DamageFunction/"

# read data
panel <- read.csv(paste0(loc_panel,"Attribution_DamageFunction_Panel.csv"))
panel$Temp2 <- (panel$Temp)**2
panel$Precip2 <- (panel$Precip)**2
panel$Temp_UDel2 <- (panel$Temp_UDel)**2
panel$Temp_20cr2 <- (panel$Temp_20cr)**2
panel$Temp_BEST2 <- (panel$Temp_BEST)**2

##### BHM specification

# first the contemporaneous regression with trends and fixed effects
# bootstrap by country

set.seed(120)
nboot <- 250
fe <- "countrynum + Year"
cl <- "0"

# set up data 
contemporaneous_coefs <- data.frame("boot"=c(1:nboot),
                                    "coef_t"=numeric(nboot),
                                    "coef_t2"=numeric(nboot))

# trends
trend_lin <- names(panel)[substr(names(panel), 1, 9)=="yi_linear"] %>%
  paste0(" + ") %>% as.list %>% do.call(paste0, .)
trend_quad <- names(panel)[substr(names(panel), 1, 12)=="yi_quadratic"] %>% 
  paste0(" + ") %>% as.list %>% do.call(paste0, .)


# formula
contemporaneous_formula <- as.formula(paste0("growth ~ Temp + Temp2 + ",
                                             trend_lin,trend_quad," Precip + Precip2",
                                             " | ",fe," | 0 | ",cl))

# bootstrap
for (n in c(1:nboot)){
  print(n)
  
  # sample by country with replacement
  isos <- unique(panel$ISO)
  isos_boot <- sample(isos,size=length(isos),replace=T)
  df_boot <- sapply(isos_boot, function(x) which(panel[,'ISO']==x))
  data_boot <- panel[unlist(df_boot),]
  
  # run model 
  mdl <- felm(contemporaneous_formula,data=data_boot)
  
  # get coefs
  contemporaneous_coefs[n,"coef_t"] <- coef(summary(mdl))["Temp","Estimate"]
  contemporaneous_coefs[n,"coef_t2"] <- coef(summary(mdl))["Temp2","Estimate"]
}

write.csv(contemporaneous_coefs,paste0(loc_out,"Attribution_Coefficients_Bootstrap_BHMSR.csv"))





# next the lagged regression
nlag <- 5
vars_to_lag <- c("Temp","Precip","Temp2","Precip2")

for (v in c(1:length(vars_to_lag))){
  var <- vars_to_lag[v]
  print(var)
  for (l in c(1:nlag)){
    panel %>% group_by(ISO) %>% 
      mutate(!!paste(var,"_lag",l,sep="") := dplyr::lag((!!as.name(var)),l)) -> panel
  }
}  

# set up data 
lagged_coefs <- data.frame("boot"=c(1:nboot),
                           "coef_t"=numeric(nboot),
                           "coef_t2"=numeric(nboot))

# set up formula
lagged_formula_initial <- paste0("growth ~ Temp + Temp2 + ",
                                 trend_lin,trend_quad," Precip + Precip2")
   
for (j in c(1:nlag)){
  lagged_formula_initial <- paste0(lagged_formula_initial," + Temp_lag",j," + Temp2_lag",j," + Precip_lag",j," + Precip2_lag",j)
}
lagged_formula <- as.formula(paste0(lagged_formula_initial," | ",fe," | 0 | ",cl))
t_name <- "Temp"
t2_name <- "Temp2"
for (j in c(1:nlag)){
  t_name <- c(t_name,paste0("Temp_lag",j))
  t2_name <- c(t2_name,paste0("Temp2_lag",j))
} 

panel <- data.frame(panel)

# bootstrap
for (n in c(1:nboot)){
  print(n)
  
  # sample by country with replacement
  isos <- unique(panel$ISO)
  isos_boot <- sample(isos,size=length(isos),replace=T)
  df_boot <- sapply(isos_boot, function(x) which(panel[,'ISO']==x))
  data_boot <- panel[unlist(df_boot),]
  
  # run model 
  mdl <- felm(lagged_formula,data=data_boot)
  
  # get coefs
  lagged_coefs[n,"coef_t"] <- sum(coef(summary(mdl))[t_name,"Estimate"])
  lagged_coefs[n,"coef_t2"] <- sum(coef(summary(mdl))[t2_name,"Estimate"])
}

write.csv(lagged_coefs,paste0(loc_out,"Attribution_Coefficients_Bootstrap_BHMLR.csv"))





##### Rich vs. poor

# poor = below-median ppp-adjusted per capita GDP in the
# first year the country enters the dataset

panel$region_time <- paste(panel$region,panel$Year,sep="_")
median_gpc_ppp <- median(panel[panel$Year==1990,"GPC_PPP_1990"],na.rm=T)
panel$rich <- as.numeric(panel$GPC_PPP_1990 > median_gpc_ppp)
panel$poor <- as.numeric(panel$GPC_PPP_1990 <= median_gpc_ppp)
panel$rp_time <- paste(panel$Year,panel$rich,sep="_")

summary(felm(as.formula("growth ~ Temp*poor | countrynum + region_time + rp_time | 0 | 0"),
             data=panel[panel$Year<=2000,]))


form <- as.formula(paste0("growth ~ Temp + Temp2 + Temp*rich + Temp2*rich + ",trend_lin,trend_quad," Precip + Precip2 | countrynum + Year | 0 | countrynum"))
summary(felm(form,data=panel))



## now bootstrap with different samples

# trends
trend_lin <- names(panel)[substr(names(panel), 1, 9)=="yi_linear"] %>%
  paste0(" + ") %>% as.list %>% do.call(paste0, .)
trend_quad <- names(panel)[substr(names(panel), 1, 12)=="yi_quadratic"] %>% 
  paste0(" + ") %>% as.list %>% do.call(paste0, .)


# formula
set.seed(120)
nboot <- 250
fe <- "countrynum + Year"
cl <- "0"
rp_formula <- as.formula(paste0("growth ~ Temp + Temp2 + ",
                                trend_lin,trend_quad," Precip + Precip2",
                                " | ",fe," | 0 | ",cl))

rp_coefs <- data.frame("boot"=c(1:nboot),
                       "coef_t_poor"=numeric(nboot),
                       "coef_t2_poor"=numeric(nboot),
                       "coef_t_rich"=numeric(nboot),
                       "coef_t2_rich"=numeric(nboot))  

for (k in c(0,1)){
  panel_rp <- panel[panel$rich==k,]
  isos <- unique(panel_rp$ISO)
  
  if(k==0){lab="poor"}else if(k==1){lab="rich"}
  print(lab)
  
  for (n in c(1:nboot)){
    print(n)
    isos_boot <- sample(isos,size=length(isos),replace=T)
    df_boot <- sapply(isos_boot, function(x) which(panel_rp[,'ISO']==x))
    data_boot <- panel_rp[unlist(df_boot),]
    mdl <- felm(rp_formula,data=data_boot)
    rp_coefs[n,paste0("coef_t_",lab)] <- coef(summary(mdl))["Temp","Estimate"]
    rp_coefs[n,paste0("coef_t2_",lab)] <- coef(summary(mdl))["Temp2","Estimate"]
  }
}
write.csv(rp_coefs,paste0(loc_out,"Attribution_Coefficients_Bootstrap_BHMRP.csv"))




##### Different temperature datasets

temp_dataset_coefs <- data.frame("coef_t_UDel"=numeric(1),
                                 "coef_t2_UDel"=numeric(1),
                                 "coef_t_20cr"=numeric(1),
                                 "coef_t2_20cr"=numeric(1),
                                 "coef_t_BEST"=numeric(1),
                                 "coef_t2_BEST"=numeric(1))

# trends
trend_lin <- names(panel)[substr(names(panel), 1, 9)=="yi_linear"] %>%
  paste0(" + ") %>% as.list %>% do.call(paste0, .)
trend_quad <- names(panel)[substr(names(panel), 1, 12)=="yi_quadratic"] %>% 
  paste0(" + ") %>% as.list %>% do.call(paste0, .)

fe <- "countrynum + Year"
cl <- "0"

temp_datasets <- c("UDel","20cr","BEST")

for (k in (1:length(temp_datasets))){
  # formula
  ds <- temp_datasets[k]
  print(ds)
  form <- as.formula(paste0("growth ~ Temp_",ds," + Temp_",ds,"2 + ",
                            trend_lin,trend_quad," Precip + Precip2",
                            " | ",fe," | 0 | ",cl))
  mdl <- felm(form,data=panel)
  temp_dataset_coefs[paste0("coef_t_",ds)] <- coef(summary(mdl))[paste0("Temp_",ds),"Estimate"]
  temp_dataset_coefs[paste0("coef_t2_",ds)] <- coef(summary(mdl))[paste0("Temp_",ds,"2"),"Estimate"]
}


write.csv(temp_dataset_coefs,paste0(loc_out,"Attribution_Coefficients_Temp_Datasets.csv"))
