import os, sys, re, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from .params import elems
from . import MAPI_KEY

import pymatgen
from pymatgen.core import Composition
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram


def echem_stab(chemsys, pickle_path, pbx_dia=None, pH=8.5, V=-1.2, conc=6e-4):
    """
    Pulls Pourbaix stabilities of all concentrations in 'concs' for elements A and B,
    then pickles the resulting dataframe as "fpre+$A+$B+'.pkl'".
    Note that the saved dataframe is not processed in any way.
    !This is a much faster implementation across the entire dataset than PourbaixStabilitiesC!


    Args:
        chemsys    a list of elements
        pH         The pH of the solution
        V          The electronic potential of the solution in volts
        fpre       The prefix of the filepath to be saved at

    Returns:
        None
    """

    failed_entries = []
    df = pd.DataFrame({})
    if pbx_dia == None:
        with MPRester(MAPI_KEY) as m:
            pourbaix_data = m.get_pourbaix_entries([*chemsys, "O", "H"])
            pbx_dia = PourbaixDiagram(
                pourbaix_data, conc_dict={elem: conc for elem in chemsys}
            )
    for entry in pbx_dia.all_entries:
        if type(entry) is pymatgen.analysis.pourbaix_diagram.PourbaixEntry:
            pbx_list = [
                entry,
            ]
        else:
            pbx_list = list(entry.entry_list)
        try:
            sub_df = pd.DataFrame(
                [
                    {
                        "V": V,
                        "pH": pH,
                        "chemsys": chemsys,
                        "formula": e.name,
                        "energy": pbx_dia.get_decomposition_energy(entry, pH=pH, V=V),
                        "entry_id": e.entry_id,
                    }
                    for e in pbx_list
                ]
            )
            sub_df["type"] = sub_df.formula.apply(
                lambda x: [
                    "ion"
                    if ("+" in x or "-" in x or "(aq)" in x)
                    else "hydroxide"
                    if "OH" in x
                    else "sulfide"
                    if "S" in x.replace("Sn", "")
                    else "oxide"
                    if "O" in x.replace("Os", "")
                    else "hydride"
                    if "H" in x.replace("Hf", "").replace("Ho", "")
                    else "solid"
                    if "(s)" in x
                    else "?"
                ][0]
            )
            df = pd.concat((df, sub_df))
        except ValueError:
            failed_entries.append(entry)
    if len(failed_entries) > 0:
        print(
            str(len(failed_entries))
            + " pourbaix entries could not be analyzed in system pH:%0.1f|V:%0.1f|"
            % (pH, V)
            + "-".join(chemsys)
        )
    if not df.empty:
        df.to_pickle(pickle_path)
    else:
        print("-".join(chemsys) + " could not be analyzed")
    return df


def echem_stabilities(*args, overwrite=False, unique_entries=True, **kwargs):
    echem_df = pull_echem_stabilities(*args, overwrite=overwrite, **kwargs)
    if unique_entries:
        echem_df["chemsys_str"] = echem_df["chemsys"].apply(str)
        echem_df = echem_df[
            echem_df.energy
            == echem_df.groupby(
                ["V", "pH", "chemsys_str", "entry_id"]
            ).energy.transform(min)
        ].drop("chemsys_str", axis=1)
    return echem_df


def binary_echem_stabilities(
    dir_path="../data/pourbaix_stabilities", file_prefix="echem_stab_", **kwargs
):
    # Convenience function to run all binary combinations of a given metal list
    kwargs.setdefault(
        "V_range",
        [
            -1.2,
        ],
    )
    kwargs.setdefault(
        "pH_range",
        [
            8.5,
        ],
    )
    kwargs.setdefault("conc", 6e-4)

    metal_pairs = []
    for idx_A in range(len(elems)):
        for idx_B in range(idx_A, len(elems)):
            elem_A = elems[idx_A]
            elem_B = elems[idx_B]
            if elem_A == elem_B:
                metal_pairs.append((elem_A,))
            else:
                metal_pairs.append((elem_A, elem_B))
    return echem_stabilities(
        metal_pairs, dir_path=dir_path, file_prefix=file_prefix, **kwargs
    )


def pull_echem_stabilities(
    chemistries,
    pH_range=[
        8.5,
    ],
    V_range=[
        -1.2,
    ],
    pH_step=None,
    V_step=None,
    conc=6e-4,
    dir_path="../data/pourbaix_stabilities",
    file_prefix="echem_stab_",
    overwrite=True,
    **kwargs
):

    # Include right-hand limits
    if (len(pH_range) == 2) and (pH_step is not None):
        pH_range[1] = pH_range[1] + 10 * np.finfo(np.float64).eps
        pHs = np.arange(*pH_range, pH_step)
    else:
        pHs = pH_range

    if (len(V_range) == 2) and (V_step is not None):
        V_range[1] = V_range[1] + 10 * np.finfo(np.float64).eps
        Vs = np.arange(*V_range, V_step)
    else:
        Vs = V_range

    pkls = [dir_path + "/" + f for f in os.listdir(dir_path)]
    df = pd.DataFrame({})

    for chem in tqdm(chemistries):
        if "X" in chem:
            chem.remove("X")
        pbx_dia = None
        for pH in pHs:
            for V in Vs:
                pickle_path = (
                    dir_path
                    + "/"
                    + file_prefix
                    + "-".join(chem)
                    + "_V_%0.2f_pH_%0.2f_conc_%i.pkl" % (V, pH, int(np.log10(conc)))
                )
                if (not overwrite) and (pickle_path in pkls):
                    df = pd.concat((df, pd.read_pickle(pickle_path)))
                else:
                    if pbx_dia == None:
                        with MPRester(MAPI_KEY) as m:
                            pourbaix_data = m.get_pourbaix_entries([*chem, "O", "H"])
                            pbx_dia = PourbaixDiagram(
                                pourbaix_data, conc_dict={elem: conc for elem in chem}
                            )
                    df = pd.concat(
                        (
                            df,
                            echem_stab(
                                pickle_path=pickle_path,
                                chemsys=chem,
                                pbx_dia=pbx_dia,
                                pH=pH,
                                V=V,
                                conc=conc,
                                **kwargs
                            ),
                        )
                    )
        pbx_dia = None
    return df


def experimental_data():
    return pd.read_pickle("../data/experimental_data.pkl")


def adsorption_energies(min_energy=False):
    ads_df = pd.read_pickle("../data/gemnet_relax_90_adsorption_energies.pkl")
    if min_energy:
        ads_df["adsorption_energy_H"] = ads_df.min_adsorption_energy
        ads_df = ads_df.drop(["min_adsorption_energy", "adsorption_energies"], axis=1)
    else:
        ads_df["adsorption_energy_H"] = ads_df.surf_adsorption_energies
        ads_df = ads_df.drop(["min_adsorption_energy", "adsorption_energies"], axis=1)
        ads_df = ads_df.explode("adsorption_energy_H")
    return ads_df
