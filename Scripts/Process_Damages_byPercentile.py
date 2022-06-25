# Processing economic damages from historical climate change and indivdiual countries
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

# data locations
loc_shp = "../Data/CountryShapefile/"
loc_gmst_fair = "../Data/FAIR/GMST/"
loc_panel = "../Data/Panel/"
loc_pop = "../Data/Population/"
loc_damages_hist = "../Data/Damages/Historical/"
loc_damages_country = "../Data/Damages/Country-Attributed/"
loc_out = "../Data/Damages/Processed/"

# get command line info
accounting = sys.argv[1] # consumption vs. territorial accounting
funcname = sys.argv[2] # damage function (e.g., BHMSR for BHM short run)
y1_shares = int(sys.argv[3]) # year 1 of emissions shares
y2_shares = int(sys.argv[4]) # year 2 of emissions shares
y1_damages = int(sys.argv[5]) # year 1 of damages calculation
y2_damages = int(sys.argv[6]) # year 2 of damages calculation

# set countries that we'll be doing this for
iso_attr = ["USA","CHN","GBR"] #"IND",

# read panel and get actual GDP data to calculate quantiles
panel = pd.read_csv(loc_panel+"Attribution_DamageFunction_Panel.csv")
iso_unique = np.array(list(sorted(set(panel["ISO"].values)))).astype(str)
panel_yrs = panel.loc[(panel["Year"]>=y1_damages)&(panel["Year"]<=y2_damages),:]
gpc = panel_yrs["GPC"].values
iso_panel = panel_yrs["ISO"].values
iso_continuous = [len(gpc[(~np.isnan(gpc)) & (iso_panel==iso)]) == len(np.arange(y1_damages,y2_damages+1,1)) for iso in iso_unique]
iso_uq_continuous = iso_unique[iso_continuous]

# slice panel data
obs_data = panel_yrs.loc[:,["Year","ISO","GPC","Temp","lnGPC","growth"]]
# calculate mean across years
obs_data_mean = obs_data.groupby("ISO").mean().reset_index().loc[:,["ISO","GPC"]]

obs_gpc = xr.DataArray(obs_data_mean.GPC.values,
                        coords=[obs_data_mean.ISO],dims=["iso"])

# calculate percentiles/quantiles
gpc_quantiles = np.array(pd.qcut(obs_gpc.values,[0,0.2,0.4,0.6,0.8,1.0],labels=False)) + 1.0
iso_quantiles = np.arange(1,5+1,1)

# now loop through emitting coujntries and quantiles
# get damages, average over countries and compute various
# metrics across uncertainty (s.d., percentiles, etc.)
# and write out

fname_str = funcname+"_"+accounting+"_shares"+str(y1_shares)+"-"+str(y2_shares)+"_"+str(y2_damages)

first_section = False
if first_section:
    for i in iso_attr:
        print(i,flush=True)
        files_list = np.array([x for x in os.listdir(loc_damages_country+i+"/") if ("global" not in x)&(fname_str+".nc" in x)&("fulldist" in x)])
        iso_files = [x[15:18] for x in files_list]
        for q in iso_quantiles:
            print(q,flush=True)

            # read in data
            iso_quantile = obs_gpc[gpc_quantiles==q].coords["iso"].values
            files_quantile = files_list[[x in iso_quantile for x in iso_files]]
            files_quantile_final = [loc_damages_country+i+"/"+x for x in files_quantile]
            damage_quantile_in = xr.open_mfdataset(files_quantile_final,concat_dim="iso")
            damage_quantile_mean = damage_quantile_in.pct_damage.mean(dim="iso").load()

            # calculate various metrics of uncertainty
            damage_ps = np.array([2.5,5,16.5,25,50,75,83.5,95,97.5]) # 66, 90, 95% CIs
            damage_qs = damage_ps/100.0
            damage_mean_i = damage_quantile_mean.mean(dim=["ensemble","member","fair_run","boot"])
            damage_sd_i = damage_quantile_mean.std(dim=["ensemble","member","fair_run","boot"])
            damage_pctiles_i = damage_quantile_mean.quantile(damage_qs,dim=["ensemble","member","fair_run","boot"])

            # concatenate/build out arrays
            if q==iso_quantiles[0]:
                damage_mean_isoattr = damage_mean_i.expand_dims("iso_quantile")
                damage_sd_isoattr = damage_sd_i.expand_dims("iso_quantile")
                damage_pctiles_isoattr = damage_pctiles_i.expand_dims("iso_quantile")
            else:
                damage_mean_isoattr = xr.concat([damage_mean_isoattr,damage_mean_i],dim="iso_quantile")
                damage_sd_isoattr = xr.concat([damage_sd_isoattr,damage_sd_i],dim="iso_quantile")
                damage_pctiles_isoattr = xr.concat([damage_pctiles_isoattr,damage_pctiles_i],dim="iso_quantile")

        damage_mean_isoattr.coords["iso_quantile"] = iso_quantiles
        damage_sd_isoattr.coords["iso_quantile"] = iso_quantiles
        damage_pctiles_isoattr.coords["iso_quantile"] = iso_quantiles

        if i==iso_attr[0]:
            damage_mean = damage_mean_isoattr.expand_dims("iso_attr")
            damage_sd = damage_sd_isoattr.expand_dims("iso_attr")
            damage_pctiles = damage_pctiles_isoattr.expand_dims("iso_attr")
        else:
            damage_mean = xr.concat([damage_mean,damage_mean_isoattr],dim="iso_attr")
            damage_sd = xr.concat([damage_sd,damage_sd_isoattr],dim="iso_attr")
            damage_pctiles = xr.concat([damage_pctiles,damage_pctiles_isoattr],dim="iso_attr")


    #print(damage_pctiles)
    damage_mean.coords["iso_attr"] = iso_attr
    damage_sd.coords["iso_attr"] = iso_attr
    damage_pctiles.coords["iso_attr"] = iso_attr

    # wrap into dataset and write out
    damage_quantile_out = xr.Dataset({"damage_mean":(["iso_attr","iso_quantile"],damage_mean),
                                      "damage_sd":(["iso_attr","iso_quantile"],damage_sd),
                                      "damage_pctiles":(["iso_attr","iso_quantile","quantile"],damage_pctiles)},
                                      coords={"iso_attr":("iso_attr",iso_attr),
                                              "iso_quantile":("iso_quantile",iso_quantiles),
                                              "quantile":("quantile",damage_qs)})

    damage_quantile_out.attrs["creation_date"] = str(datetime.datetime.now())
    damage_quantile_out.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    damage_quantile_out.attrs["variable_description"] = "selected attributable GDP changes grouped by average GDP quintiles"
    damage_quantile_out.attrs["created_from"] = os.getcwd()+"/Process_Damages_byPercentile.py"
    damage_quantile_out.attrs["ensemble_details"] = "Perturbed-parameter ensemble for carbon cycle and TCR/ECS"
    damage_quantile_out.attrs["emissions_subtraction"] = "Leave-one-out country attribution using shares of CEDS emissions"

    fname_out = loc_out+"attributable_damages_byincome_"+fname_str+".nc"
    damage_quantile_out.to_netcdf(fname_out,mode="w")
    print(fname_out,flush=True)



###### a different version
gpc_quintiles = xr.DataArray(np.array(pd.qcut(obs_gpc.values,[0,0.2,0.4,0.6,0.8,1.0],labels=False))+1.0,
                            coords=[obs_data_mean.ISO],dims=["iso"])
iso_quintiles = np.arange(1,5+1,1)
attr_quintiles = np.arange(1,5+1,1)
iso = gpc_quintiles.iso.values

fname_str = funcname+"_"+accounting+"_shares"+str(y1_shares)+"-"+str(y2_shares)+"_"+str(y2_damages)

attr_damage_avg_mean = xr.DataArray(np.zeros((len(attr_quintiles),len(iso_quintiles))),
                                coords=[attr_quintiles,iso_quintiles],
                                dims=["attr_quintile","damaged_quintile"])
attr_damage_avg_sd = xr.DataArray(np.zeros((len(attr_quintiles),len(iso_quintiles))),
                                coords=[attr_quintiles,iso_quintiles],
                                dims=["attr_quintile","damaged_quintile"])
attr_damage_sum_mean = xr.DataArray(np.zeros((len(attr_quintiles),len(iso_quintiles))),
                                coords=[attr_quintiles,iso_quintiles],
                                dims=["attr_quintile","damaged_quintile"])
attr_damage_sum_sd = xr.DataArray(np.zeros((len(attr_quintiles),len(iso_quintiles))),
                                coords=[attr_quintiles,iso_quintiles],
                                dims=["attr_quintile","damaged_quintile"])


for i in np.arange(0,len(attr_quintiles),1):
    print(attr_quintiles[i],flush=True)
    iso_attrs = iso[gpc_quintiles==attr_quintiles[i]]
    for j in np.arange(0,len(iso_quintiles),1):
        print(iso_quintiles[j],flush=True)
        q_countries = iso[gpc_quintiles==iso_quintiles[j]]
        for iso_attr in iso_attrs:
            print(iso_attr,flush=True)
            files_list = np.array([x for x in os.listdir(loc_damages_country+iso_attr+"/") if ("global" not in x)&(fname_str+".nc" in x)&("fulldist" in x)])
            iso_files = [x[15:18] for x in files_list]
            files_quintile = files_list[[x in q_countries for x in iso_files]]
            files_quintile_final = [loc_damages_country+iso_attr+"/"+x for x in files_quintile]
            damages_quintile_in = xr.open_mfdataset(files_quintile_final,concat_dim="iso")
            damage_quintile = damages_quintile_in.pct_damage.load()
            if iso_attr==iso_attrs[0]:
                damage_quintile_isoattr = damage_quintile.expand_dims("iso_attr")
            else:
                damage_quintile_isoattr = xr.concat([damage_quintile_isoattr,damage_quintile],dim="iso_attr")

        attr_damage_avg_mean[i,j] = damage_quintile_isoattr.mean(dim=["iso_attr","iso"]).mean(dim=["ensemble","member","fair_run","boot"])
        attr_damage_avg_sd[i,j] = damage_quintile_isoattr.mean(dim=["iso_attr","iso"]).std(dim=["ensemble","member","fair_run","boot"])
        attr_damage_sum_mean[i,j] = damage_quintile_isoattr.sum(dim="iso_attr").mean(dim="iso").mean(dim=["ensemble","member","fair_run","boot"])
        attr_damage_sum_sd[i,j] = damage_quintile_isoattr.sum(dim="iso_attr").mean(dim="iso").std(dim=["ensemble","member","fair_run","boot"])

attr_damage_out = xr.Dataset({"attr_damage_avg_mean":(["attr_quintile","damaged_quintile"],attr_damage_avg_mean),
                              "attr_damage_avg_sd":(["attr_quintile","damaged_quintile"],attr_damage_avg_sd),
                              "attr_damage_sum_mean":(["attr_quintile","damaged_quintile"],attr_damage_sum_mean),
                              "attr_damage_sum_sd":(["attr_quintile","damaged_quintile"],attr_damage_sum_sd)},
                                  coords={"attr_quintile":("attr_quintile",attr_quintiles),
                                          "damaged_quintile":("damaged_quintile",iso_quintiles)})

attr_damage_out.attrs["creation_date"] = str(datetime.datetime.now())
attr_damage_out.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
attr_damage_out.attrs["variable_description"] = "attributable GDP changes grouped by average GDP quintiles"
attr_damage_out.attrs["created_from"] = os.getcwd()+"/Process_Damages_byPercentile.py"
attr_damage_out.attrs["ensemble_details"] = "Perturbed-parameter ensemble for carbon cycle and TCR/ECS"
attr_damage_out.attrs["emissions_subtraction"] = "Leave-one-out country attribution using shares of CEDS emissions"

fname_out = loc_out+"attributable_damages_byincome_"+fname_str+"_v2.nc"
attr_damage_out.to_netcdf(fname_out,mode="w")
print(fname_out,flush=True)
