# Processing data on damages from historical climate change and indivdiual countries
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
sig_test = sys.argv[7] # significance test, either ks or t

# Shapefile
shp = gp.read_file(loc_shp)
iso_shp = shp.ISO3.values

# Number of FAIR runs
n_fair_runs = 250

# read in FAIR GMST to get iso_attr
y1_fair = 1850
y2_fair = 2014
fair_gmst = xr.open_dataset(loc_gmst_fair+"FAIR_GMST_PPE"+str(n_fair_runs)+"_"+accounting+"_shares"+str(y1_shares)+"-"+str(y2_shares)+"_"+str(y1_fair)+"-"+str(y2_fair)+".nc")
iso_attr = fair_gmst.coords["iso_attr"].values

# type of significance test
#sig_test = "t" #ks

# now get attributable damages for all of those countries
# read in each country-by-country file to be able to split out
# losses and gains

if sig_test=="ks":
    fname_str = funcname+"_"+accounting+"_shares"+str(y1_shares)+"-"+str(y2_shares)+"_"+str(y1_damages)+"-"+str(y2_damages)
elif sig_test=="t":
    fname_str = funcname+"_"+accounting+"_shares"+str(y1_shares)+"-"+str(y2_shares)+"_"+str(y1_damages)+"-"+str(y2_damages)+"_ttest"
else:
    print("ERROR -- significance test should be either ks or t")
    sys.exit()

for i in iso_attr:
    print(i)
    files_list = [x for x in os.listdir(loc_damages_country+i+"/") if ("global" not in x)&(fname_str+".nc" in x)&("fulldist" not in x)]
    iso = [x[15:18] for x in files_list]
    for j in iso:
        fname = i+"-attributed_"+j+"_gdp_damages_"+fname_str+".nc"
        damage_to_country = xr.open_dataset(loc_damages_country+i+"/"+fname)
        if j == iso[0]:
            dollar_damages = damage_to_country.dollar_damage_mean
        else:
            dollar_damages = xr.concat([dollar_damages,damage_to_country.dollar_damage_mean],dim="iso")

    if i == iso_attr[0]:
        global_damage_attribution = dollar_damages.expand_dims("iso_attr")
    else:
        global_damage_attribution = xr.concat([global_damage_attribution,dollar_damages],dim="iso_attr")
global_damage_attribution.coords["iso_attr"] = iso_attr

# calculate benefits and losses and sum
attributable_benefits = global_damage_attribution.where(global_damage_attribution>=0,0.0).sum(dim="iso")
attributable_losses = global_damage_attribution.where(global_damage_attribution<=0,0.0).sum(dim="iso")

# wrap into dataset and write out
attr_damages_out = xr.Dataset({"attributable_benefits":(["iso_attr","time"],attributable_benefits),
                                "attributable_losses":(["iso_attr","time"],attributable_losses)},
                                coords={"iso_attr":("iso_attr",attributable_benefits.iso_attr),
                                        "time":("time",attributable_benefits.time)})

attr_damages_out.attrs["creation_date"] = str(datetime.datetime.now())
attr_damages_out.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
attr_damages_out.attrs["variable_description"] = "attributable GDP changes (both benefits and losses), summed across affected countries"
attr_damages_out.attrs["created_from"] = os.getcwd()+"/Process_Damages.py"
attr_damages_out.attrs["ensemble_details"] = "Perturbed-parameter ensemble for carbon cycle and TCR/ECS"
attr_damages_out.attrs["emissions_subtraction"] = "Leave-one-out country attribution using shares of CEDS emissions"

fname_out = loc_out+"global_attributable_benefits_losses_"+fname_str+".nc"
attr_damages_out.to_netcdf(fname_out,mode="w")
print(fname_out,flush=True)
