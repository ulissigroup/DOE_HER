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
    """Pulls Pourbaix stabilities of all concentrations in 'concs' for elements A and B,
    then pickles the resulting dataframe as "fpre+$A+$B+'.pkl'".
    Note that the saved dataframe is not processed in any way.

    Args:
        chemsys (List[str]): A list of elements
        pickle_path (str): location to pickle the output
        pbx_dia (pymatgen.analysis.pourbaix_diagram.PourbaixDiagram, optional): electrochemical system to evaluate the stability of. Defaults to None.
        pH (float, optional): pH of interest. Defaults to 8.5.
        V (float, optional): voltage of interest. Defaults to -1.2.
        conc (float, optional): ion concentration of interest. Defaults to 6e-4.

    Returns:
        pd.core.frame.DataFrame: electrochemical system and stability of all entries in that system.
    """
    """


    Args:
        chemsys    a list of elements
        pH         The pH of the solution
        V          The electronic potential of the solution in volts
        fpre       The prefix of the filepath to be saved at

    Returns:
        pd.core.frame.DataFrame: electrochemical system and the stabilities of all entries in that system.
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
    """Convenience function to get electrochemical stabilities of all binary combinations.

    Args:
        dir_path (str, optional): Path of the directory where electrochemical stability data is stored. Defaults to "../data/pourbaix_stabilities".
        file_prefix (str, optional): Prepended to the chemical system to name electrochemical stability files. Defaults to "echem_stab_".

    Returns:
        pd.core.frame.DataFrame: the electrochemical system and the stability of all entries in that system.
    """
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
    """Get electrochemical entries and stability data from the Materials Project API.

    Args:
        chemistries (List[List[str]]): Metal systems of interest
        pH_range (List[float], optional): pH maximum and minimum of interest. Defaults to [ 8.5, ].
        V_range (List[float], optional): Voltage maximum and minimum of interest. Defaults to [ -1.2, ].
        pH_step (float_, optional): Difference between consecutive pHs to pull data for. If None, sample the first and second point of pH_range. Defaults to None.
        V_step (List[float], optional): Difference between consecutive voltages to pull data for. If None, sample the first and second point of V_range. Defaults to None.
        conc (float, optional): Ion concentrations to assume. Defaults to 6e-4.
        dir_path (str, optional): Location of directory to store data in. Defaults to "../data/pourbaix_stabilities".
        file_prefix (str, optional): Prepended to the metal system to form individual file names. Defaults to "echem_stab_".
        overwrite (bool, optional): If True, overwrite existing files. Defaults to True.

    Returns:
        pd.core.frame.DataFrame: The electrochemical systems and stabilities of each entry.
    """
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
    """Load experimental data

    Returns:
        pd.core.frame.DataFrame: Experimental data for water/TEOA systems.
    """
    return pd.read_pickle("../data/experimental_data.pkl")


def adsorption_energies(min_energy=False):
    """Load hydrogen adsorption energies for relevant chemical systems as predicted by Gemnet after relaxing for 90 steps.

    Args:
        min_energy (bool, optional): Return the minimum adsorption energy on each surface. Defaults to False.

    Returns:
        pd.core.frame.DataFrame: Adsorption energy predictions.
    """
    ads_df = pd.read_pickle("../data/gemnet_relax_90_adsorption_energies.pkl")
    if min_energy:
        ads_df["adsorption_energy_H"] = ads_df.min_adsorption_energy
        ads_df = ads_df.drop(["min_adsorption_energy", "adsorption_energies"], axis=1)
    else:
        ads_df["adsorption_energy_H"] = ads_df.adsorption_energies
        ads_df = ads_df.drop(["min_adsorption_energy", "adsorption_energies"], axis=1)
        ads_df = ads_df.explode("adsorption_energy_H")
    return ads_df
