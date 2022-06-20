# Economic damages from historical climate change and indivdiual countries
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

# Dependencies
import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
from rasterio import features
from affine import Affine
import geopandas as gp
import descartes
from scipy import stats
import psutil
import warnings

# Data locations
loc_shp = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/ProcessedCountryShapefile/"
loc_gmst_fair = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/FAIR/GMST/"
loc_panel = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/Panel/"
loc_pop = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/Population/"
loc_damagefunc = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/DamageFunction/"
loc_pattern = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/PatternScaling/"
loc_out_tdiff_global = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/FAIR/TempDifference/Historical/"
loc_out_tdiff_countries = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/FAIR/TempDifference/Country-Attributed/"
loc_out_damages_hist = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/Damages/Historical/"
loc_out_damages_countries = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/Damages/Country-Attributed/"

# get command line info
accounting = sys.argv[1] # consumption vs. territorial accounting
funcname = sys.argv[2] # damage function (e.g., BHMSR for BHM short run)
y1_shares = int(sys.argv[3]) # year 1 of emissions shares
y2_shares = int(sys.argv[4]) # year 2 of emissions shares
y1_damages = int(sys.argv[5]) # year 1 of damages calculation
y2_damages = int(sys.argv[6]) # year 2 of damages calculation
iso_loop_section = sys.argv[7] # portion of the country loop

# Shapefile
shp = gp.read_file(loc_shp)
iso_shp = shp.ISO3.values

# Number of FAIR runs
n_fair_runs = 250

# read in FAIR GMST
y1_fair = 1850
y2_fair = 2014
fair_gmst = xr.open_dataset(loc_gmst_fair+"FAIR_GMST_PPE"+str(n_fair_runs)+"_"+accounting+"_shares"+str(y1_shares)+"-"+str(y2_shares)+"_"+str(y1_fair)+"-"+str(y2_fair)+".nc")
gmst_fair_all = fair_gmst.t_global
gmst_fair_subtract = fair_gmst.t_subtract
gmst_fair_pic = fair_gmst.t_pic
iso_attr = fair_gmst.coords["iso_attr"]
fair_runs = fair_gmst.coords["fair_run"]

# calc difference from time-varying natural simulation
# same with hist minus hist-nat when doing the pattern scaling
gmst_diff_fair_all = gmst_fair_all - gmst_fair_pic
gmst_diff_fair_subtract = gmst_fair_subtract - gmst_fair_pic

# Read in pattern scaling coefficients
pattern_coefs = xr.open_dataarray(loc_pattern+"GCM_linear_country_scaling_coefficients_smoothed.nc")
ensembles = pattern_coefs.coords["ensemble"].values
members = pattern_coefs.coords["member"].values
iso = pattern_coefs.coords["iso"].values
order = len(pattern_coefs.coords["order"].values) - 1

# predict historical temperatures
def calc_predicted_t(c,coefs,order):
    if order == 1:
        predicted = coefs[1,:,:,:] + (c*coefs[0,:,:,:])
    if order == 2:
        predicted = coefs[2,:,:,:] + (c*coefs[1,:,:,:]) + ((c**2) * coefs[0,:,:,:])
    return(predicted)

t_hist_pred = calc_predicted_t(gmst_diff_fair_all,pattern_coefs,order)

# Set years for damages
years_damages = np.arange(y1_damages,y2_damages,1)
if y1_damages < y1_shares:
    print("ERROR: You are trying to attribute damages for years prior to when emissions shares were calculated!",flush=True)
    sys.exit()

# Read in panel and limit to countries with continuous GDP data
panel = pd.read_csv(loc_panel+"Attribution_DamageFunction_Panel.csv")
iso_unique = np.array(list(sorted(set(panel["ISO"].values)))).astype(str)
panel_yrs = panel.loc[(panel["Year"] >= y1_damages) & (panel["Year"]<=y2_damages)]
gpc = panel_yrs["GPC"].values
iso_panel = panel_yrs["ISO"].values
iso_continuous = [len(gpc[(~np.isnan(gpc)) & (iso_panel==iso)]) == len(np.arange(y1_damages,y2_damages+1,1)) for iso in iso_unique]
iso_uq_continuous = iso_unique[iso_continuous]

# set rich/poor variable
## this might give us a setting-with-copy warning but that can be ignored
median_gpc_ppp = np.nanmedian(panel.loc[panel.Year==1990,"GPC_PPP_1990"].values)
panel_yrs.loc[:,"rich"] = (panel_yrs.loc[:,"GPC_PPP_1990"].values > median_gpc_ppp).astype(int)

# Damage function coefficients
lr_lag_length = 5

if funcname == "BHMSR":
    damagefunc = pd.read_csv(loc_damagefunc+"Attribution_TempCoefs_Bootstrap_Contemporaneous.csv",index_col=0)
elif funcname == "BHMLR":
    damagefunc = pd.read_csv(loc_damagefunc+"Attribution_TempCoefs_Bootstrap_Lag"+str(lr_lag_length)+".csv",index_col=0)
elif funcname == "BHMRP":
    damagefunc = pd.read_csv(loc_damagefunc+"Attribution_Coefficients_Bootstrap_BHMRP.csv",index_col=0)
else:
    print("ERROR: function should be BHMSR, BHMLR, or BHMRP")
    sys.exit()

if funcname in ["BHMSR","BHMLR"]:
    b1 = damagefunc.loc[:,"coef_t"].values
    b2 = damagefunc.loc[:,"coef_t2"].values
    nboot = len(b1)
    boot = np.arange(1,nboot+1,1)
    def damage_function(t,rich):
        func = (t * b1) + ((t**2) * b2)
        return(func)

#print(b1)
#print(b2)
#print(np.std(b1))
#print(np.std(b2))
#sys.exit()


if funcname=="BHMRP":
    b1_rich = damagefunc.loc[:,"coef_t_rich"].values
    b2_rich = damagefunc.loc[:,"coef_t2_rich"].values
    b1_poor = damagefunc.loc[:,"coef_t_poor"].values
    b2_poor = damagefunc.loc[:,"coef_t2_poor"].values
    nboot = len(b1_rich)
    boot = np.arange(1,nboot+1,1)
    def damage_function(t,rich):
        if rich==1:
            func = (t * b1_rich) + ((t**2) * b2_rich)
        elif rich==0:
            func = (t * b1_poor) + ((t**2) * b2_poor)
        elif np.isnan(rich):
            func = np.nan
        return(func)


# construct t obs
years = np.arange(y1_damages,y2_damages+1,1)

t_obs = xr.DataArray(np.full((len(iso_uq_continuous),len(years)),np.nan),
                    coords=[iso_uq_continuous,years],
                    dims=["iso","time"])
for ii in np.arange(0,len(iso_uq_continuous),1):
    for yy in np.arange(0,len(years),1):
        t_obs.loc[iso_uq_continuous[ii],years[yy]] = panel.loc[(panel["ISO"]==iso_uq_continuous[ii])&(panel["Year"]==years[yy]),"Temp"].values[0]

obs_data = panel_yrs.loc[:,["Year","ISO","GPC","Temp","lnGPC","growth","rich"]]

# construct population data
pop_data = pd.read_csv(loc_pop+"WPP2019_Country_Population.csv")
pop_data_num = pop_data["Country code"].values
num_shp = shp.UN.values

iso_pop = xr.DataArray(np.full((len(iso_uq_continuous),len(years)),np.nan),
                       coords=[iso_uq_continuous,years],
                       dims=["iso","time"])
for ii in np.arange(0,len(iso_uq_continuous),1):
    for yy in np.arange(0,len(years),1):
        # pop data is in units of thousands
        pop_str = pop_data.loc[pop_data_num==num_shp[iso_shp==iso_uq_continuous[ii]][0],str(years[yy])].values
        if len(pop_str) != 0:
            pop_float = float("".join((pop_str[0]).strip().split()))
            iso_pop[ii,yy] = pop_float*1000
        else:
            iso_pop[ii,yy] = np.nan

# helpful functions
def create_growth_arrays(delta_growth_xr,ensembles,members,fair_runs,years,boot):
    cf_gdp = xr.DataArray(np.full(delta_growth_xr.values.shape,np.nan),
                         coords=[ensembles,members,fair_runs,years,boot],
                         dims=["ensemble","member","fair_run","time","boot"])
    cf_growth = xr.DataArray(np.full(delta_growth_xr.values.shape,np.nan),
                         coords=[ensembles,members,fair_runs,years,boot],
                         dims=["ensemble","member","fair_run","time","boot"])
    return([cf_gdp,cf_growth])


# create actual_gdp and actual_growth arrays
actual_gdp = xr.DataArray(np.full((len(years),len(iso_uq_continuous)),np.nan),
                     coords=[years,iso_uq_continuous],
                     dims=["time","iso"])
rich = xr.DataArray(np.full(len(iso_uq_continuous),np.nan),
                    coords=[iso_uq_continuous],dims=["iso"])

for jj in np.arange(0,len(iso_uq_continuous),1):
    obs_gdp = obs_data.loc[obs_data["ISO"]==iso_uq_continuous[jj],"GPC"].values
    #obs_growth = obs_data.loc[obs_data["ISO"]==iso_uq_continuous[jj],"growth"].values
    actual_gdp[:,jj] = obs_gdp
    #actual_growth[:,jj] = obs_growth
    rich_bool = obs_data.loc[obs_data["ISO"]==iso_uq_continuous[jj],"rich"].values[0]
    rich[jj] = rich_bool

actual_growth1 = actual_gdp.diff(dim="time",n=1)
actual_frac_growth = actual_growth1/(actual_gdp.loc[:(y2_damages-1),:].values)
growth_nans = xr.DataArray(np.full((1,len(iso_uq_continuous)),np.nan),
							coords=[[y1_damages],iso_uq_continuous],
							dims=["time","iso"])
actual_growth = xr.concat([growth_nans,actual_frac_growth],dim="time")

# Loop through each country and attribute damages to them. For a select few
# (e.g., the US and China, write out country-level attributable damages),
# but for the rest of them, just aggregate to the global level and output those totals.

iso_global_attr_orig = fair_gmst.coords["iso_attr"].values
iso_countrylevel_attr = ["USA","CHN","IND","GBR","RUS"]

# change attribution list to get US and China (and others) first
iso_attr_first = ["USA","CHN","JPN","KOR","DEU","GBR","CAN"]
attr_country_indices = [iso_attr_orig not in iso_attr_first for iso_attr_orig in iso_global_attr_orig]
iso_global_attr_orig_2 = iso_global_attr_orig[attr_country_indices]
iso_global_attr = np.append(iso_attr_first,iso_global_attr_orig_2)

# run damages calculation

print(iso_global_attr)
if iso_loop_section == "first":
    if funcname=="BHMLR":
        begin_country = "ETH"
        iso_range = np.arange(list(iso_global_attr).index(begin_country),np.floor(len(iso_global_attr)/3),1)
    elif (funcname=="BHMRP")&(accounting=="territorial")&(y1_damages==1990):
        begin_country = "SLV"
        iso_range = np.arange(list(iso_global_attr).index(begin_country),np.floor(len(iso_global_attr)/3),1)
    else:
        iso_range = np.arange(0,np.floor(len(iso_global_attr)/3),1)
    #iso_range = np.arange(list(iso_global_attr).index(begin_country),np.floor(len(iso_global_attr)/3),1)
    #iso_range = np.arange(list(iso_global_attr).index(begin_country),list(iso_global_attr).index(end_country)+1,1)
elif iso_loop_section == "second":
    if funcname=="BHMLR":
        begin_country = "MNG"
        iso_range = np.arange(list(iso_global_attr).index(begin_country),np.floor(len(iso_global_attr)/3)*2,1)
    elif (funcname=="BHMSR")&(accounting=="territorial")&(y1_damages==1990):
        begin_country = "HUN" # for ttest version
        iso_range = np.arange(list(iso_global_attr).index(begin_country),np.floor(len(iso_global_attr)/3)*2,1)
    else:
        iso_range = np.arange(np.floor(len(iso_global_attr)/3),np.floor(len(iso_global_attr)/3)*2,1)
    #iso_range = np.arange(list(iso_global_attr).index(begin_country),np.floor(len(iso_global_attr)/3)*2,1)
    #iso_range = np.arange(list(iso_global_attr).index(begin_country),list(iso_global_attr).index(end_country)+1,1)
elif iso_loop_section == "third":
    if funcname=="BHMLR":
        begin_country = "BWA"
        iso_range = np.arange(list(iso_global_attr).index(begin_country),len(iso_global_attr),1)
    elif (funcname=="BHMRP")&(accounting=="territorial")&(y1_damages==1990):
        begin_country = "JEY"
        iso_range = np.arange(list(iso_global_attr).index(begin_country),len(iso_global_attr),1)
    else:
        iso_range = np.arange(np.floor(len(iso_global_attr)/3)*2,len(iso_global_attr),1)
    #iso_range = np.arange(list(iso_global_attr).index(begin_country),len(iso_global_attr),1)
    #iso_range = np.arange(list(iso_global_attr).index(begin_country),list(iso_global_attr).index(end_country)+1,1)
elif iso_loop_section == "all":
    iso_range = np.arange(0,len(iso_global_attr),1)
else:
    print("ERROR: iso_loop_section incorrectly defined!",flush=True)
    sys.exit()

# suppress annoying warning for mean of empty slice
warnings.filterwarnings("ignore",message="Mean of empty",
                        category=RuntimeWarning)

for ii in iso_range:

    iso_to_attr = iso_global_attr[int(ii)]
    print(iso_to_attr+", country #"+str(list(iso_global_attr).index(iso_to_attr)),flush=True)

    print("calculating temperature change",flush=True)
    t_subtract_pred = calc_predicted_t(gmst_diff_fair_subtract.loc[iso_to_attr,:,:],pattern_coefs,order)
    tdiff_hist_isoattr = t_hist_pred.loc[:,:,iso_uq_continuous,years,:] - t_subtract_pred.loc[:,:,iso_uq_continuous,years,:]
    tdiff_hist_pic = t_subtract_pred.loc[:,:,iso_uq_continuous,years,:] #- t_pic_co2eq.loc[:,:,iso_uq_continuous,years,:]

    # calculate counterfactual temperatures
    t_cf_iso = t_obs - tdiff_hist_isoattr.transpose("ensemble","member","fair_run","iso","time")
    t_cf_pic = t_obs - tdiff_hist_pic.transpose("ensemble","member","fair_run","iso","time")

    # add bootstrap coordinates to enable damage function calculation
    t_cf_iso_boot = t_cf_iso.expand_dims(boot=nboot)
    t_cf_iso_boot.coords["boot"] = np.arange(1,nboot+1,1)
    t_cf_iso_boot = t_cf_iso_boot.transpose("ensemble","member","fair_run","time","iso","boot")

    t_cf_pic_boot = t_cf_pic.expand_dims(boot=nboot)
    t_cf_pic_boot.coords["boot"] = np.arange(1,nboot+1,1)
    t_cf_pic_boot = t_cf_pic_boot.transpose("ensemble","member","fair_run","time","iso","boot")

    t_obs_boot = t_obs.expand_dims(boot=t_cf_iso_boot.coords["boot"])
    t_obs_boot = t_obs_boot.transpose("time","iso","boot")

    del([t_subtract_pred,tdiff_hist_isoattr,t_cf_iso,t_cf_pic])

    # apply damage function and calculate growth change
    print("applying damage function and calculating damages for individual nations",flush=True)
    for jj in np.arange(0,len(iso_uq_continuous),1):
        iso_damage = iso_uq_continuous[jj]
        print(iso_damage,flush=True)

        loc_out_country_attr_damages = loc_out_damages_countries+iso_to_attr+"/"
        if os.path.exists(loc_out_country_attr_damages)==False:
            os.mkdir(loc_out_country_attr_damages)

        # applying damage function
        func_obs = damage_function(t_obs_boot.loc[:,iso_damage,:],rich.loc[iso_damage].values)
        func_cf_subtract = damage_function(t_cf_iso_boot.loc[:,:,:,:,iso_damage,:],rich.loc[iso_damage].values)
        func_cf_pic = damage_function(t_cf_pic_boot.loc[:,:,:,:,iso_damage,:],rich.loc[iso_damage].values)
        deltagrowth_subtract = func_cf_pic - func_cf_subtract
        deltagrowth_pic = func_cf_pic - func_obs
        #delta_growth = func_cf_iso - func_obs
        del([func_cf_subtract,func_obs])

        # function to make this easier
        def calc_counterfactual_gdp(iso,actual_growth,actual_gdp,delta_growth,ens=ensembles,mem=members,fair_runs=fair_runs,yrs=years,boot=boot):
            cf_gdp, cf_growth = create_growth_arrays(delta_growth,ens,mem,fair_runs,yrs,boot)
            cf_growth[:,:,:,:,:] = (actual_growth.loc[:,iso] + delta_growth).transpose("ensemble","member","fair_run","time","boot")
            for yy in np.arange(1,len(yrs),1):
                if yy == 1:
                    cf_gdp_year = actual_gdp.loc[yrs[yy-1],iso]
                else:
                    cf_gdp_year = cf_gdp.loc[:,:,:,yrs[yy-1],:]
                cf_gdp.loc[:,:,:,yrs[yy],:] = cf_gdp_year.values + (cf_growth.loc[:,:,:,yrs[yy],:].values * cf_gdp_year.values)

            return(cf_gdp)

        # calculate counterfactual GDP per capita arrays
        cf_gdp_pic = calc_counterfactual_gdp(iso_damage,actual_growth,actual_gdp,deltagrowth_pic)
        cf_gdp_isosubtract = calc_counterfactual_gdp(iso_damage,actual_growth,actual_gdp,deltagrowth_subtract)

        # attributable change in GDP per capita
        total_gdp_change = (actual_gdp.loc[:,iso_damage] - cf_gdp_pic).transpose("ensemble","member","fair_run","time","boot")
        gdp_change_isosubtract = (actual_gdp.loc[:,iso_damage] - cf_gdp_isosubtract).transpose("ensemble","member","fair_run","time","boot")
        attributable_gdp_change = total_gdp_change - gdp_change_isosubtract

        # attributable change in percent
        gdp_damage_pic = (((actual_gdp.loc[:,iso_damage] - cf_gdp_pic)/cf_gdp_pic)*100).transpose("ensemble","member","fair_run","time","boot")
        gdp_damage_isosubtract = (((actual_gdp.loc[:,iso_damage] - cf_gdp_isosubtract)/cf_gdp_isosubtract)*100).transpose("ensemble","member","fair_run","time","boot")
        attributable_damage_pct = gdp_damage_pic - gdp_damage_isosubtract

        #print("calculating significance",flush=True)
        # test significance
        sig_test = "t" # "ks" #"t"
        alpha = 0.05
        def ks_p(x,y):
            ks, p = stats.ks_2samp(x,y)
            return(p)
        def ttest_p(x,y):
            t, p = stats.ttest_ind(x,y,nan_policy="omit")
            return(p)

        # significance test
        dist1 = gdp_damage_pic.stack(uncert=("ensemble","member","fair_run","boot"))
        dist2 = gdp_damage_isosubtract.stack(uncert=("ensemble","member","fair_run","boot"))
        if sig_test=="ks":
            p = xr.apply_ufunc(ks_p,dist1,dist2,vectorize=True,input_core_dims=[["uncert"],["uncert"]])
        elif sig_test=="t":
            p = xr.apply_ufunc(ttest_p,dist1,dist2,vectorize=True,input_core_dims=[["uncert"],["uncert"]])
        else:
            print("ERROR: significance test incorrectly specified")
            sys.exit()
        sig = (p < alpha).astype(int)

        #print("adding to global total",flush=True)
        # sum dollar gains and losses -- previously, weighting by GDP per capita relative to world
        # but not doing that here
        # Bernoulli-Nash/Cobb-Douglas weights
        # see fankhauser et al., 1997
        attributable_dollar_change_country = attributable_gdp_change*iso_pop.loc[iso_damage,:]
        equity_wgts = 1.0 # (actual_gdp.mean(dim="iso"))/actual_gdp.loc[:,iso_damage]
        attributable_damage_dollar = attributable_dollar_change_country * equity_wgts * sig

        # add to total
        # but split out benefits and damages
        if jj == 0:
            attributable_dollar_change = attributable_damage_dollar
            attributable_benefits = attributable_damage_dollar.where(attributable_damage_dollar>0,0.0)
            attributable_losses = attributable_damage_dollar.where(attributable_damage_dollar<0,0.0)
        else:
            attributable_dollar_change = attributable_dollar_change + attributable_damage_dollar.where(~np.isnan(attributable_damage_dollar),0)
            attributable_benefits = attributable_benefits + attributable_damage_dollar.where((attributable_damage_dollar>0)&(~np.isnan(attributable_damage_dollar)),0.0)
            attributable_losses = attributable_losses + attributable_damage_dollar.where((attributable_damage_dollar<0)&(~np.isnan(attributable_damage_dollar)),0.0)

        #print("writing out data",flush=True)

        pctiles = np.array([2.5,5,16.5,25,50,75,83.5,95,97.5]) # 66, 90, 95% CIs
        qs = pctiles/100.0

        # damages *without* the country in question
        pct_damage_subtract_mean = gdp_damage_isosubtract.mean(dim=["ensemble","member","fair_run","boot"])
        pct_damage_subtract_pctiles = gdp_damage_isosubtract.quantile(qs,dim=["ensemble","member","fair_run","boot"])

        # write out historical damages to have them for later
        # just for one iso_attr country because they're the same
        if sig_test=="ks":
            fname_str = funcname+"_"+accounting+"_shares"+str(y1_shares)+"-"+str(y2_shares)+"_"+str(y1_damages)+"-"+str(y2_damages)
        elif sig_test=="t":
            fname_str = funcname+"_"+accounting+"_shares"+str(y1_shares)+"-"+str(y2_shares)+"_"+str(y1_damages)+"-"+str(y2_damages)+"_ttest"

        if iso_to_attr == "USA":
            actual_gdp_iso = actual_gdp.loc[:,iso_damage]
            gdp_damage_mean = gdp_damage_pic.mean(dim=["ensemble","member","fair_run","boot"])
            gdp_damage_sd = gdp_damage_pic.std(dim=["ensemble","member","fair_run","boot"])
            gdp_damage_pctiles = gdp_damage_pic.quantile(qs,dim=["ensemble","member","fair_run","boot"])
            cf_gdp_mean = cf_gdp_pic.mean(dim=["ensemble","member","fair_run","boot"])
            hist_damage_ds = xr.Dataset({"actual_gdp":(["time"],actual_gdp_iso),
                                        "cf_gdp_mean":(["time"],cf_gdp_mean),
                                        "gdp_damage_mean":(["time"],gdp_damage_mean),
                                        "gdp_damage_std":(["time"],gdp_damage_sd),
                                        "gdp_damage_pctiles":(["quantile","time"],gdp_damage_pctiles)},
                                        coords={"ensemble":(["ensemble"],ensembles),
                                                "member":(["member"],members),
                                                "fair_run":(["fair_run"],fair_runs),
                                                "time":(["time"],years),
                                                "boot":(["boot"],boot),
                                                "quantile":(["quantile"],qs)})

            hist_damage_ds.attrs["creation_date"] = str(datetime.datetime.now())
            hist_damage_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
            hist_damage_ds.attrs["variable_description"] = "actual gdp, counterfactual gdp, and percent difference"
            hist_damage_ds.attrs["created_from"] = os.getcwd()+"/Calculate_Damages.py"
            hist_damage_ds.attrs["ensemble_details"] = "Perturbed-parameter ensemble for carbon cycle and TCR/ECS"
            hist_damage_ds.attrs["emissions_subtraction"] = "Leave-one-out country attribution using shares of CEDS emissions"

            fname_out = loc_out_damages_hist+iso_damage+"_gdp_damages_hist_preindustrial_"+fname_str+".nc"
            hist_damage_ds.to_netcdf(fname_out,mode="w")
            print(fname_out,flush=True)


        # write out country-level damages -- include percentiles and standard deviations
        attributable_damage_pct = attributable_damage_pct.transpose("ensemble","member","fair_run","boot","time")

        # pct damages
        pct_damage_mean = attributable_damage_pct.mean(dim=["ensemble","member","fair_run","boot"])
        pct_damage_mean = pct_damage_mean.expand_dims("iso")
        pct_damage_pctiles = attributable_damage_pct.quantile(qs,dim=["ensemble","member","fair_run","boot"])
        pct_damage_pctiles = pct_damage_pctiles.expand_dims("iso")
        pct_damage_sd_total = attributable_damage_pct.std(dim=["ensemble","member","fair_run","boot"])
        pct_damage_sd_boot = attributable_damage_pct.mean(dim=["ensemble","member","fair_run"]).std(dim=["boot"])
        pct_damage_sd_fair = attributable_damage_pct.mean(dim=["ensemble","member","boot"]).std(dim=["fair_run"])
        pct_damage_sd_iv = attributable_damage_pct.loc["CESM1-SFLE",:,:,:].mean(dim=["fair_run","boot"]).std(dim=["member"])
        pct_damage_sd_mdl = attributable_damage_pct.loc["CMIP6",:,:,:].mean(dim=["fair_run","boot"]).std(dim=["member"])

        # actual dollar damages
        dollar_damage_mean = attributable_damage_dollar.mean(dim=["ensemble","member","fair_run","boot"])
        dollar_damage_mean = dollar_damage_mean.expand_dims("iso")
        dollar_damage_sd = attributable_damage_dollar.std(dim=["ensemble","member","fair_run","boot"])
        dollar_damage_sd = dollar_damage_sd.expand_dims("iso")

        # dataset
        countrylevel_damage_out = xr.Dataset({"pct_damage_mean":(["iso","time"],pct_damage_mean),
                                            "pct_damage_pctiles":(["quantile","iso","time"],pct_damage_pctiles.transpose("quantile","iso","time")),
                                            "pct_damage_subtract_mean":(["time"],pct_damage_subtract_mean),
                                            "pct_damage_subtract_pctiles":(["quantile","time"],pct_damage_subtract_pctiles),
                                            "pct_damage_sd_total":(["time"],pct_damage_sd_total),
                                            "pct_damage_sd_boot":(["time"],pct_damage_sd_boot),
                                            "pct_damage_sd_fair":(["time"],pct_damage_sd_fair),
                                            "pct_damage_sd_iv":(["time"],pct_damage_sd_iv),
                                            "pct_damage_sd_mdl":(["time"],pct_damage_sd_mdl),
                                             "dollar_damage_mean":(["iso","time"],dollar_damage_mean),
                                             "dollar_damage_sd_total":(["iso","time"],dollar_damage_sd)},
                                             coords={"ensemble":("ensemble",ensembles),
                                                    "member":("member",members),
                                                    "fair_run":("fair_run",fair_runs),
                                                    "time":("time",years),
                                                    "boot":("boot",boot),
                                                    "iso":("iso",pct_damage_mean.iso),
                                                    "quantile":(["quantile"],qs)})

        countrylevel_damage_out.attrs["creation_date"] = str(datetime.datetime.now())
        countrylevel_damage_out.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
        countrylevel_damage_out.attrs["variable_description"] = "attributable GDP damages at the country level"
        countrylevel_damage_out.attrs["created_from"] = os.getcwd()+"/Calculate_Damages.py"
        countrylevel_damage_out.attrs["ensemble_details"] = "Perturbed-parameter ensemble for carbon cycle and TCR/ECS"
        countrylevel_damage_out.attrs["emissions_subtraction"] = "Leave-one-out country attribution using shares of CEDS emissions"

        fname_out = loc_out_country_attr_damages+iso_to_attr+"-attributed_"+iso_damage+"_gdp_damages_"+fname_str+".nc"
        countrylevel_damage_out.to_netcdf(fname_out,mode="w")
        print(fname_out,flush=True)

        # write out entire distribution of country-level damages
        # we'll just slice it to 2014 because that captures all the previously accumulated damage
        attributable_damage_pct_out = (attributable_damage_pct*sig).loc[:,:,:,:,y2_damages].expand_dims("iso")
        attributable_damage_pct_out.name = "pct_damage"
        attributable_damage_pct_out.attrs["creation_date"] = str(datetime.datetime.now())
        attributable_damage_pct_out.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
        attributable_damage_pct_out.attrs["variable_description"] = "Country-level GDP changes"
        attributable_damage_pct_out.attrs["created_from"] = os.getcwd()+"/Calculate_Damages.py"
        attributable_damage_pct_out.attrs["ensemble_details"] = "Perturbed-parameter ensemble for carbon cycle and TCR/ECS"
        attributable_damage_pct_out.attrs["emissions_subtraction"] = "Leave-one-out country attribution using shares of CEDS emissions"

        fname_out = loc_out_country_attr_damages+iso_to_attr+"-attributed_"+iso_damage+"_gdp_damages_fulldist_"+fname_str+".nc"
        attributable_damage_pct_out.to_netcdf(fname_out,mode="w")
        print(fname_out,flush=True)

    # write out total global damages
    attributable_dollar_change = attributable_dollar_change.transpose("ensemble","member","fair_run","boot","time")
    attributable_losses = attributable_losses.transpose("ensemble","member","fair_run","boot","time")
    attributable_benefits = attributable_benefits.transpose("ensemble","member","fair_run","boot","time")
    global_damages_ds = xr.Dataset({"attributable_dollar_change":(["ensemble","member","fair_run","boot","time"],attributable_dollar_change),
                                    "attributable_benefits":(["ensemble","member","fair_run","boot","time"],attributable_benefits),
                                    "attributable_losses":(["ensemble","member","fair_run","boot","time"],attributable_losses)},
                                         coords={"ensemble":("ensemble",ensembles),
                                                "member":("member",members),
                                                "fair_run":("fair_run",fair_runs),
                                                "time":("time",years),
                                                "boot":("boot",boot)})

    #attributable_dollar_change = attributable_dollar_change.expand_dims("iso")
    #global_damages_ds.name = "attributable_damages"
    global_damages_ds.attrs["creation_date"] = str(datetime.datetime.now())
    global_damages_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    global_damages_ds.attrs["variable_description"] = "Global total GDP changes, gains, and losses"
    global_damages_ds.attrs["created_from"] = os.getcwd()+"/Calculate_Damages.py"
    global_damages_ds.attrs["ensemble_details"] = "Perturbed-parameter ensemble for carbon cycle and TCR/ECS"
    global_damages_ds.attrs["emissions_subtraction"] = "Leave-one-out country attribution using shares of CEDS emissions"

    fname_out = loc_out_country_attr_damages+iso_to_attr+"-attributed_global_gdp_damages_"+fname_str+".nc"
    global_damages_ds.to_netcdf(fname_out,mode="w")
    print(fname_out,flush=True)
