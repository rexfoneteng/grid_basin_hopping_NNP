#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# ==============================================================
# Copyright(c) 2017-, Po-Jen Hsu (clusterga@gmail.com)
# Copyright(c) 2018-, Teh Soon (tehsoonts@gmail.com)
# See the README file in the top-level directory for license.
# ==============================================================
# Creation Time : 20170627 11:40:01
# ==============================================================
from collections import OrderedDict
from os.path import basename, isfile

from numpy import array, exp, float_power, hstack, pi, sqrt
from scipy.optimize import least_squares

from array_tools import eigen_dict
from const_tools import (Avogadro, Bohr_radius, atomic_mass, atomic_number, ev,
                         number2atom, wavenumber)
from file_tools import escaped_file_name
from list_tools import join_2d_list, sort_index, sort_with_index, exp_sample_id_list
from os_tools import file_dir, full_path, job_cmd, mktemp, shell_cmd
from time_tools import delta_time, now
from xyz_tools import (Xyz, g09_inp2xyz, g09_out2xyz, mol_pattern,
                       molecule_stat, xyz2list, xyz_type)

inv_bohr_radius_ang = float("{:.7f}".format(1.0 / Bohr_radius["angstrom"]))

default_header_str = "%nproc=1\n%mem=20Gb\n# pm6 opt empiricaldispersion=gd3\n\nremark here\n\n0 1"

iop_dict = {"input_orientation": "IOP(2/9=2000)"}

Default_header = {}


def out2gibbs(*args, **kwargs):
    """
    Returns:
        Gibbs free energy at standard condition (~298K): (float)
    Args:
        input: gaussian output filename (str)
        pref: preference dict with above parameters as keys
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref = update_pref(args, kwargs, pref)
    gibbs_info = \
        shell_cmd("grep -a 'Sum of electronic and thermal Free Energies' {}"
                  .format(pref["input"]))
    gibbs_free_eng = None
    if gibbs_info:
        gibbs_free_eng = float(gibbs_info[0].split()[-1])
    return result(gibbs_free_eng, pref)


def out2part(*args, **kwargs):
    """
    Returns:
        part_dict: (dict, {"pos_list":, "freq_list":})
    Args:
        input: gaussian output filename (str)
        pref: preference dict with above parameters as keys
    Note:
        This function is for Dr. Cheng-Chau's part_func
    """
    pref = OrderedDict()
    pref["input"] = None
    pref = update_pref(args, kwargs, pref)
    xyz_list = xyz2list(pref["input"])
    configuration_list = []
    convert_factor = 1.0 / ev["cm^-1"] * 1000.
    for xyz in xyz_list:
        configuration_list.append([xyz[1],
                                   xyz[2],
                                   xyz[3],
                                   atomic_mass[xyz[0]],
                                   xyz[0]])
    freq_list = []
    wave_number_list = out2freq(pref["input"])
    for wave_number in wave_number_list:
        if wave_number >= 0:
            wave_number_flag = "r"
        else:
            wave_number_flag = "i"
        freq_list.append([wave_number * convert_factor,
                          wave_number,
                          wave_number_flag])
    part_dict = {
        "pos_list": configuration_list,
        "freq_list": freq_list,
    }
    return result(part_dict, pref)


def xyz2inp(*args, **kwargs):
    """
    Returns:
        0: if succeeded
    Args:
        input : xyz/inp/out_file_name (str), list of list, or list of str
        output: Gaussian input file (str)
        header: header_file or header_string (str)
        use_chk_file: True will use output filename as chk_file
        atom_type_list: a new atom_type_list to replace the original atom_type
            (list of str)
        append_info: additional information after the Xyz coordinate (str)
        use BSSE: True will enable counterpoise parameters (bool)
        BSSE_charge_atom_num_list: list of atom numbers to determine the
            charged molecules (list of int)
        charged_pattern: if the mol_type is found, it will give charge 1 (str)
        pattern_list: list of molecular type for chaging charge and
            multiplicity (list of str)
        charge_list: list of charge corresponding to the pattern_list
            (list of int)
        multiplicity_list: list of multiplicity corresponding to the
            pattern_list (list of int)
        pref: preference includes all parameters (dict)
    Note:
        If "#" is in header, it will be treated as header string.\n\n

        chk_file="output" will use the output name as chk_file.\n\n

        atom_type_list will be useful for isotop calculation.\n\n

        If "counterpoise" parameter is in the header file, use_BSSE will be
        automatically enabled.
    """
    pref = OrderedDict()
    pref["input"] = [
        "H 0.075853 0.002082 -0.003417",
        "O -0.625132 1.192093 -0.438473",
        "H -0.651382 1.264230 -1.415388",
    ]
    pref["output"] = "0.inp"  # or new_header.txt if input = None
    pref["header"] = "header.txt"  # or header string
# "%NprocShared=6\n\
# %mem=8Gb\n\
# # b3lyp/6-31+G* opt=(maxcycles=100)\n\
# \n\
# remark here\n\
# \n\
# 0 1\n"
    pref["use_chk_file"] = False
    pref["atom_type_list"] = None
    pref["append_info"] = None
    pref["use_BSSE"] = False
    pref["BSSE_charged_atom_num_list"] = []
    pref["pattern_list"] = []
    pref["charge_list"] = []
    pref["multiplicity_list"] = []
    pref = update_pref(args, kwargs, pref)
    xyz_list = xyz2list(pref["input"])
    if pref["atom_type_list"]:
        tmp_xyz_list = []
        for I0, new_atom_type in enumerate(pref["atom_type_list"]):
            tmp_xyz_list.append(tuple([new_atom_type,
                                       xyz_list[I0][1],
                                       xyz_list[I0][2],
                                       xyz_list[I0][3]]))
        xyz_list = tmp_xyz_list
    # For the case output = xyz_list
    with open(pref["output"], "w") as inp_obj:
        if pref["use_chk_file"]:
            inp_obj.write("%chk={}.chk\n".
                          format(basename(pref["output"]).split(".")[0]))
        tmp_header_str = header_str(pref["header"])
        if pref["pattern_list"]:
            molecule_pattern = mol_pattern(xyz_list,
                                           flatten=True)
            for I1, pattern in enumerate(pref["pattern_list"]):
                if pattern in molecule_pattern:
                    charge = pref["charge_list"][I1]
                    multiplicity = pref["multiplicity_list"][I1]
                    print("Detect {}: Use charge/multiplicity= {}/{} for {}"
                          .format(pattern,
                                  charge,
                                  multiplicity,
                                  pref["output"]))
                    tmp_header_str_list = tmp_header_str.split("\n")
                    tmp_header_str_list[-2] = "{} {}".format(charge,
                                                             multiplicity)
                    tmp_header_str = "\n".join(tmp_header_str_list)
                    break

        if "counterpoise" in tmp_header_str.lower():
            print("Detect counterpoise parameter. Set \033[1;36muse_BSSE\033[1;33m=\033[1;32mTrue\033[0m")
            pref["use_BSSE"] = True
            mol_xyz_list = molecule_stat(pref["input"])["mol_xyz_list"]
            old_cp_num = tmp_header_str.split("poise=")[-1].split()[0]
            tmp_header_str = tmp_header_str.replace("poise={}".format(old_cp_num),
                                                    "poise={}".format(len(mol_xyz_list)))
        elif pref["use_BSSE"]:
            mol_xyz_list = molecule_stat(pref["input"])["mol_xyz_list"]
            tmp_header_list = tmp_header_str.rstrip().split("\n")
            for I0, tmp_str in enumerate(tmp_header_list):
                if "#" in tmp_str:
                    tmp_header_list[I0] = tmp_str.rstrip() + " counterpoise={}".format(len(mol_xyz_list))
                    print("Set \033[1;36m{}\033[0m".format(tmp_header_list[I0]))
            tmp_header_str = "\n".join(tmp_header_list)
        write_xyz_list = []
        if pref["use_BSSE"]:
            print("Detect counterpoise, please check the charge and the multiplicity")
            print("or BSSE_charged_atom_num_list")
            BSSE_charge_list = []
            for I0, mol_xyz in enumerate(mol_xyz_list):
                atom_num = len(mol_xyz)
                if atom_num in pref["BSSE_charged_atom_num_list"]:
                    BSSE_charge_list.append("1 1")
                else:
                    BSSE_charge_list.append("0 1")
                for xyz in mol_xyz:
                    write_xyz_list.append("{}(Fragment={}) {:.6f} {:.6f} {:.6f}\n"
                                          .format(xyz[0], I0, xyz[1], xyz[2], xyz[3]))
            BSSE_charge_str = " ".join(BSSE_charge_list)
            tmp_header_str = tmp_header_str.rstrip() + " " + BSSE_charge_str + "\n"
        else:
            for xyz in xyz_list:
                write_xyz_list.append("{} {:.6f} {:.6f} {:.6f}\n"
                                      .format(xyz[0], xyz[1], xyz[2], xyz[3]))
        inp_obj.write(tmp_header_str)
        for write_xyz in write_xyz_list:
            inp_obj.write(write_xyz)
        inp_obj.write("\n")
        if pref["append_info"]:
            inp_obj.write(pref["append_info"])
    return result(0, pref)


def inp2xyz(*args, **kwargs):
    """
    Returns:
        0: if succeeded
    Args:
        output: xyz_filename (str)
        input: gaussian input filename (str)
        info: content of info line (str, default="")
        write_flag: "a" will append to the output xyz filename (str)
        info: info_line (str)
        pref: preference dict with above parameters as keys
    Note:
        To record frame_id, source_file, or other information,
        use info.
    """
    pref = OrderedDict()
    pref["output"] = "0.xyz"
    pref["input"] = "0.inp"
    pref["write_flag"] = "w"
    pref["info"] = ""
    pref = update_pref(args, kwargs, pref)
    xyz_list = xyz2list(pref["input"])
    atom_num = len(xyz_list)
    with open(pref["output"], pref["write_flag"]) as xyz_obj:
        xyz_obj.write("{}\n".format(atom_num))
        if "\n" in pref["info"]:
            xyz_obj.write(pref["info"])
        else:
            xyz_obj.write("{}\n".format(pref["info"]))
        for I0 in range(atom_num):
            xyz_obj.write("{} {:.6f} {:.6f} {:.6f}\n".
                          format(xyz_list[I0][0],
                                 xyz_list[I0][1],
                                 xyz_list[I0][2],
                                 xyz_list[I0][3]))
    return result(0, pref)


def out2xyz(*args, **kwargs):
    """
    Returns:
        0: if succeeded
    Args:
        input: gaussian output file name (str)
        output: output xyz file name (str)
        write_flag: "a" will append to the output xyz file (str)
        info: info line of the xyz (str)
        pot_key: will record scf_eng in xyz_info after the pot_key (str)
        zpe_key: will record zpe in xyz_info after the zpe_key (str)
        freq_key: will record freq_list_str with comma after the freq_key (str)
        inten_key: will record inten_list_str with comma after the inten_key
            (str)
        pref: preference dict with above parameters as keys
    Note:
        output="input" will use input.split(".")[0]+".xyz" as output file name

        Assign frame_id and other information to "info"
        If pot_key/zpe_key/freq_key/inten_key are assigned and not None or "",
        it will try to search the corresponding information and record them
        after each key.\n\n

        Maximal number of characters of the info_line is 999. More characters
        will cause error in some visualization software.\n\n

        It the information is missing, it will not be recorded in the xyz info
        line.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["output"] = "use_input"
    pref["write_flag"] = "w"
    pref["info"] = ""
    pref["pot_key"] = "eng="
    pref["zpe_key"] = "zpe="
    pref["freq_key"] = None   # or "freq="
    pref["inten_key"] = None  # or "inten="
    pref = update_pref(args, kwargs, pref)
    xyz_list = xyz2list(pref["input"])
    atom_num = len(xyz_list)
    if "\n" in pref["info"]:
        info = pref["info"].split("\n")[0]
    else:
        info = pref["info"]
    if pref["pot_key"]:
        scf_eng = out2eng(pref["input"])
        if scf_eng:
            info += " {} {}".format(pref["pot_key"], scf_eng)
    if pref["zpe_key"]:
        zpe = out2zpe(pref["input"])
        if zpe:
            info += " {} {}".format(pref["zpe_key"], zpe)
    if pref["freq_key"]:
        freq_list = out2freq(pref["input"])
        if freq_list:
            freq_str_list = []
            for freq in freq_list:
                freq_str_list.append(str(freq))
            info += " {} {}".format(pref["freq_key"],
                                    ",".join(freq_str_list))
    if pref["inten_key"]:
        inten_list = out2inten(pref["input"])
        if inten_list:
            inten_str_list = []
            for inten in inten_list:
                inten_str_list.append(str(inten))
            info += " {} {}".format(pref["inten_key"],
                                    ",".join(inten_str_list))
    if pref["output"] in ("input", "use_input"):
        output = pref["input"].split(".")[0] + ".xyz"
    else:
        output = pref["output"]
    if len(info) > 999:
        print("Warning! info line exceed 999 characters")
        print("Some visualization software may fail")
    print("Write xyz to {}".format(output))
    with open(output, pref["write_flag"]) as xyz_obj:
        xyz_obj.write("{}\n".format(atom_num))
        xyz_obj.write("{}\n".format(info))
        for I0 in range(atom_num):
            xyz_obj.write("{} {:.6f} {:.6f} {:.6f}\n".
                          format(xyz_list[I0][0],
                                 xyz_list[I0][1],
                                 xyz_list[I0][2],
                                 xyz_list[I0][3]))
    return result(0, pref)


def out2force(*args, **kwargs):
    """
    Returns:
        force_list: (list of tuple in Hartree/A)
    Args:
        input: Gaussian output filename (str)
        frame_list: list of opt steps to be extracted (list or "all")
        force_multiply_factor: default is 1.8897261 to convert the force unit
            of a Gaussian output file from Hartree/Bohr to Hartree/Angstrom
            (float)
        pref: preference dict with above parameters as keys
    Note:
        force_list would be:
            [(6.567e-06, 3.3e-07, 6.329e-06),
             (1.651e-06, -9.39e-07, -1.0234e-05),
             (2.351e-06, -2.813e-06, 4.32e-07)]

        Original force unit in the Gaussian output file is Hartree/Bohr_radius.
        To convert it to Hartree/A, forces will be automatically multiplied by
        1/Bohr_radius_in_Angstrom or 1.0 / 0.5291772105638411 by default.
        To avoid the conversion, set force_multiply_factor to 1.0.\n\n

        If frame_list == "all", it will extract all the opt steps
        (list of force_list).\n\n

        If only one frame is queried, will generate a force_list (list of list)

        frame_list can be [-1, -2, -3]

        frame_list will always be sorted and duplicated number will be removed.
    """
    pref = OrderedDict()
    pref["input"] = "gaussian.out"
    pref["frame_list"] = [-1]
    pref["force_multiply_factor"] = inv_bohr_radius_ang
    pref = update_pref(args, kwargs, pref)
    escaped_file = escaped_file_name(pref["input"])
    atom_num_info = shell_cmd("grep -a 'NAtoms=' {} -m 1"
                              .format(escaped_file))
    if not atom_num_info:
        test_line = shell_cmd(
            "grep -a ' ---------------------------------------------------------------------$' {} -m 3 -n | awk -F':' '{{print $1}}'"
            .format(escaped_file))
        first_line = int(test_line[0])
        second_line = int(test_line[1])
        third_line = int(test_line[2])
        if second_line - first_line == 3:
            atom_num = third_line - second_line - 1
        if not atom_num:
            print("Error! Cannot parse atom number")
            return result(1, pref)
    else:
        atom_num = int(atom_num_info[0].split()[1])
    force_line_list = shell_cmd("grep -a 'Forces (Hartrees/Bohr)' {} -n | awk -F':' '{{print $1}}'"
                                .format(escaped_file))
    force_list = []
    frame_list = []
    if force_line_list:
        total_frame = len(force_line_list)
        if pref["frame_list"] == "all":
            frame_list = list(range(total_frame))
        else:
            for frame in pref["frame_list"]:
                if frame < 0:
                    if (total_frame + frame) >= 0:
                        frame_list.append(total_frame + frame)
                elif frame < total_frame:
                    frame_list.append(frame)
            frame_list = list(set(frame_list))
            frame_list.sort()
        prev_line = 0
        with open(pref["input"], "r") as input_obj:
            for frame in frame_list:
                force_line = int(force_line_list[frame]) + 2
                for I0 in range(force_line - prev_line):
                    next(input_obj)
                frame_force_list = []
                for I0 in range(atom_num):
                    format_line = next(input_obj)
                    # atomic_number = str(int(format_line[8:16]))
                    f_x = float(format_line[23:38])
                    f_y = float(format_line[38:53])
                    f_z = float(format_line[53:68])
                    # _, atomic_number, f_x, f_y, f_z = next(input_obj).split()
                    frame_force_list.append((f_x * pref["force_multiply_factor"],
                                             f_y * pref["force_multiply_factor"],
                                             f_z * pref["force_multiply_factor"],))
                    # print(atomic_number, f_x, f_y, f_z)
                force_list.append(frame_force_list)
                prev_line = force_line + atom_num
    if len(force_list) == 1:
        return result(force_list[0], pref)
    else:
        return result(force_list, pref)


def freq2zpe(*args, **kwargs):
    """
    Returns:
        zpe: (float in Hartree)
    Args:
        freq_list: wavenumber list in cm^-1 from out2freq (list of float)
        eng_unit: from const_tools.wavenumber["zpe_xxx"]
            (str, default="zpe_hartree")
        pref: preference dict with above parameters as keys
    Note:
        The real zero-point "corrected" energy should add the SCF energy to this
        energy (the sum of electronic and zero-point energy)

        In xyz file, the keyword "zpe" refers to the sum of electronic and
        zero-point energy (the so-call zero-point corrected energy)

        In Gaussian output files, the Frequencies are acutally wavenumbers
        in cm^-1. The real frequency (Hz) = 2*pi*c(cm/s)*wavenumber(cm^-1)
    """
    pref = OrderedDict()
    pref["freq_list"] = []
    pref["eng_unit"] = "zpe_hartree"
    pref = update_pref(args, kwargs, pref)
    zpe = 0.0
    for freq in pref["freq_list"]:
        zpe += freq
    return result(wavenumber[pref["eng_unit"]] * zpe, pref)


def out2fullfrag_cmd(*args, **kwargs):
    """
    Returns:
        read_cmd: cmd input for obtaining the full fragment calculation
            output
    Args:
        input: gaussian output_file (str)
        pref: preference which includes all parameters (dict)
    Note:
        will return "cat gaussian.out" if counterpoise calculation is not done
            in the calculation regardless of the use of fragment
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref = update_pref(args, kwargs, pref)
    escaped_file = escaped_file_name(pref["input"])
    # check for fragments in counterpoise calculation, does not include bsse eng!!!
    if shell_cmd("grep -a -m 1 'Counterpoise: doing' {}".format(escaped_file)):
        full_calc_num = shell_cmd("grep -a -n -m 2 'Counterpoise: doing' {} | awk -F: '{{print $1}}'"
                                  .format(escaped_file))
        read_cmd = "sed '{},{}!d' {}".format(full_calc_num[0].strip(),
                                             full_calc_num[1].strip(),
                                             escaped_file)
        frag_num = shell_cmd("grep -a ' in fragment ' {} | tail -1 | awk '{{print $9}}'"
                             .format(escaped_file))[0].strip().rstrip(".")
        print(
            "\033[94mCounterpoise calculation for {} fragments detected\033[0m".format(frag_num))
    else:
        read_cmd = "cat {}".format(escaped_file)

    return result(read_cmd, pref)


def out2scf(*args, **kwargs):
    """
    Returns:
        gaussian_scf_eng: (float or None in Hartree)
    Args:
        input: gaussian output_file (str)
        frame_list: list of opt steps to be extracted (list or "all")
        pref: preference which includes all parameters (dict)
    Note:
        returns the Hartree-Fock energies for ab-initio methods (e.g. MP2), but
            is the same as running out2eng for DFT.\n\n

        If frame_list == "all", it will extract all the opt steps
        (list of force_list).\n\n

        If only one frame is queried, will generate a force_list (list of list)

        frame_list can be [-1, -2, -3]

        frame_list will always be sorted and duplicated number will be removed.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["frame_list"] = [-1]
    pref = update_pref(args, kwargs, pref)

    read_cmd = out2fullfrag_cmd(pref=pref)["return"]

    scf_list = shell_cmd("{} | grep -a 'SCF Done:  E(' | awk '{{print $5}}'"
                         .format(read_cmd))

    frame_list = []
    total_frame = len(scf_list)
    if pref["frame_list"] == "all":
        frame_list = list(range(total_frame))
    else:
        for frame in pref["frame_list"]:
            if frame < 0:
                if (total_frame + frame) >= 0:
                    frame_list.append(total_frame + frame)
            elif frame < total_frame:
                frame_list.append(frame)
        frame_list = list(set(frame_list))
        frame_list.sort()
    tmp_scf_list = []
    for frame_id in frame_list:
        tmp_scf_list.append(float(scf_list[frame_id].replace("D", "E")))

    if len(tmp_scf_list) == 1:
        return result(tmp_scf_list[0], pref)
    else:
        return result(tmp_scf_list, pref)


def out2corr(*args, **kwargs):
    """
    Returns:
        gaussian_corr_eng: (float or None in Hartree)
    Args:
        input: gaussian output_file (str)
        pref: preference which includes all parameters (dict)
    Note:
        returns the correlation energies for MPN methods
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref = update_pref(args, kwargs, pref)

    read_cmd = out2fullfrag_cmd(pref=pref)["return"]

    if shell_cmd("grep -a -m 1 -E 'UMP[0-9]' {}"
                 .format(escaped_file_name(pref["input"]))):
        # get EN for MPN approximation
        corr = shell_cmd("{} | grep -a -E 'UMP[0-9]' | tail -1 | awk '{{print $3}}'"
                         .format(read_cmd))

    try:
        corr = float(corr[-1].replace("D", "E"))
    except (IndexError, NameError):
        print("will only work for MPN, please update if out2corr is supposed to work on {}"
              .format(pref["input"]))
        corr = None

    return result(corr, pref)


def scan_id_list(*args, **kwargs):
    """
    Returns:
        List of the last frame of the scanning points: (list of int)
    Args:
        input: gaussian output_file (str)
        pref: preference dict with above parameters as keys
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref = update_pref(args, kwargs, pref)
    scan_str_list = shell_cmd("grep -a 'scan point' {} |awk '{{print $13}}'"
                              .format(pref["input"]))
    tmp_scan_id_list = []
    if scan_str_list:
        prev_str = scan_str_list[0]
        for I0, scan_str in enumerate(scan_str_list):
            if scan_str != prev_str:
                tmp_scan_id_list.append(I0 - 1)
            prev_str = scan_str
    return result(tmp_scan_id_list, pref)


def out2eng(*args, **kwargs):
    """
    Returns:
        gaussian_eng: (float or list of float in Hartree)
    Args:
        input: gaussian output_file (str)
        frame_list: list of opt steps to be extracted (list or "all")
        pref: preference which includes all parameters (dict)
    Note:
        Will return only the electronic energy, i.e. no ZPE, BSSE correction
            included.\n\n

        If frame_list == "all", it will extract all the opt steps
        (list of eng).\n\n

        If only one frame is queried, will generate an eng (float)

        frame_list can be [-1, -2, -3]

        frame_list will always be sorted and duplicated number will be removed.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["frame_list"] = [-1]
    pref = update_pref(args, kwargs, pref)

    # check for fragments in counterpoise calculation
    read_cmd = out2fullfrag_cmd(pref=pref)["return"]
    escaped_file = escaped_file_name(pref["input"])
    found_corr = shell_cmd("grep -a -m 1 'E(CORR)' {}"
                           .format(escaped_file))
    found_mp2 = shell_cmd("grep -a -m 1 -E 'EUMP[0-9] =' {}"
                          .format(escaped_file))
    found_scf = shell_cmd("grep -a -m 1 -E 'SCF Done:  E' {}"
                          .format(escaped_file))
    # read the final energy
    if found_corr:
        # get couple cluster energy
        if shell_cmd("grep -a -m 1 'CCSD(T)= ' {}"
                     .format(escaped_file)):
            # triple excitation
            eng_list = shell_cmd("{} | grep -a 'CCSD(T)= ' | awk '{{ print $2 }}'"
                                 .format(read_cmd))
            print("\033[94mCCSD(T) energy detected\033[0m")
        else:
            # single and double excitation
            eng_list = shell_cmd("{} | grep -a 'E(CORR)' | awk '{{ print $4 }}'"
                                 .format(read_cmd))
            print("\033[94mCCSD energy detected\033[0m")
    elif found_mp2:
        # get (HF + E2 + ... + EN) for MPN approximation
        eng_list = shell_cmd("{} | grep -a -E 'UMP[0-9] =' | awk '{{ print $6 }}'"
                             .format(read_cmd))
        if shell_cmd("grep -a 'requencies --' {}".format(escaped_file)):
            eng_list = eng_list[:-1]
        print("\033[94mMP{} energy detected\033[0m"
              .format(found_mp2[0].split()[3][-1]))
    elif found_scf:
        # get Hartree-Fock / DFT energy
        eng_list = shell_cmd("{} | grep -a 'SCF Done:  E' | awk '{{print $5}}'"
                             .format(read_cmd))
        # if eng_list:
        #    print("\033[94mSCF energy detected\033[0m")

        # in mp2 energy, 0.22D+02 is given instead of 0.22E+02
    else:
        eng_list = shell_cmd("{} | grep -a 'Energy=  ' | grep -a -v 'correction' | awk '{{print $2}}'"
                             .format(read_cmd))
    frame_list = []
    total_frame = len(eng_list)
    if pref["frame_list"] == "all":
        frame_list = list(range(total_frame))
    else:
        for frame in pref["frame_list"]:
            if frame < 0:
                if (total_frame + frame) >= 0:
                    frame_list.append(total_frame + frame)
            elif frame < total_frame:
                frame_list.append(frame)
        frame_list = list(set(frame_list))
        frame_list.sort()
    tmp_eng_list = []
    for frame_id in frame_list:
        tmp_eng_list.append(float(eng_list[frame_id].replace("D", "E")))
    eng_list = tmp_eng_list

    if len(eng_list) == 1:
        return result(eng_list[0], pref)
    else:
        return result(eng_list, pref)


def out2cbs(*args, **kwargs):
    """
    Returns:
        extrapolated_eng: (float or dict)
    Args:
        input: gaussian output_file (list of str)
        full_output: show Hartree-Fock and correlation energy limit separately
            with dictionary key "HF" and "Corr", will only show the sum if true
            (bool, default=False)
        no_sanitize: check whether input is correct, only activate this if have
            false positive (bool, default=False)
        method: "pow", "pow2", "exp", "mix"  (str, default="mix")
        cardinal_num: cardinal numbers of basis set used in gaussian output_file
            (required only for "pow", "pow2" and "mix" method, default=[2, 3, 4])
        x:
        a:
        pref: preference which includes all parameters, see note for method
            specific pref (dict)
    Note:
        should have 3 gaussian_output files (2 can be used when method is either
            "pow", "pow2", "exp") of the same coordinates but have basis set with
            different but consecutive cardinal numbers (i.e. cc-PVXZ, X=[N, N+2])
            arranged in increasing order
        ------
        Method references:
        - pow: same as L3 extrapolation functional in MOLPRO [Helgaker et al.]: https://doi.org/10.1063/1.473863
        - pow2: same as LH3 extrapolation functional in MOLPRO [Helgaker et al.]: https://doi.org/10.1063/1.473863
        - exp: two-point linear and three-point non-linear exponential
            extrapolations by [Halkier et al.]: https://doi.org/10.1016/S0009-2614(99)00179-7
        - mix: mixed gaussian/exponential extrapolation by [Peterson et al.]: https://doi.org/10.1063/1.466884

    """
    pref = OrderedDict()
    pref["input"] = ["0.out", "1.out", "2.out"]
    pref["full_output"] = False
    pref["no_sanitize"] = False
    pref["method"] = "mix"
    pref["cardinal_num"] = [2, 3, 4]
    pref["x"] = 3
    pref["a"] = 1.63
    pref = update_pref(args, kwargs, pref)

    eng_lim = []
    hf_list = []
    xyz_list = []
    corr_list = []
    for out in pref["input"]:
        hf_list.append(out2scf(out))
        corr_list.append(out2corr(out))
        import os
        import sys
        sys.stdout = open(os.devnull, 'w')
        xyz_list.append(sorted(xyz2list(out)))
        sys.stdout = sys.__stdout__

    # sanity check
    try:
        two_params_method = ["pow", "pow2", "exp"]
        if pref["no_sanitize"]:
            print("No sanity check, make sure this is intended!!!")
        elif len(pref["input"]) == 2 and pref["method"] not in two_params_method:
            raise Exception("Only methods in {} can accept 2 gaussian output file".format(
                two_params_method))
        elif len(pref["input"]) not in [2, 3]:
            # 3 input
            raise IndexError("Should have exactly 3 gaussian output file")
        elif xyz_list[0] != xyz_list[1] != xyz_list[2]:
            # same structure
            raise Exception("The structure of the outputs is not the same")
        elif sorted(hf_list, reverse=True) != hf_list:
            # basis set not in order
            raise Exception(
                "The order of gaussian output is not in increasing basis set")
    except Exception as e:
        print(
            "set the no_sanitize=True to ignore and proceed with extrapolation (may fail)")
        raise Exception(e)

    if pref["method"] == "pow":
        for eng_list in [hf_list, corr_list]:
            eng_lim.append(
                least_squares(
                    lambda x, n, eng:
                    x[0] + (x[1] * float_power(n, -pref["x"])) - eng,
                    args=[pref["cardinal_num"],
                          eng_list], x0=[0, 0]).x[0])
    elif pref["method"] == "pow2":
        for eng_list in [hf_list, corr_list]:
            eng_lim.append(
                least_squares(
                    lambda x, n, eng:
                    x[0] + (x[1] * float_power(n + 0.5, -pref["x"])) - eng,
                    args=[array(pref["cardinal_num"]),
                          eng_list], x0=[0, 0]).x[0])
    elif pref["method"] == "exp":
        if len(hf_list) == 3:
            for eng_list in [hf_list, corr_list]:
                eng_lim.append((eng_list[0] * eng_list[2] - eng_list[1]**2) /
                               (eng_list[0] + eng_list[2] - (2 * eng_list[1])))
                # b = (eng_list[0] - eng_lim[-1] * (1 - exp(1)) -
                     # eng_list[1] * exp(1)) / (exp(-1) * (1 - exp(-2)))
        else:
            exponent = exp(-pref["a"])
            for eng_list in [hf_list, corr_list]:
                eng_lim.append((eng_list[1] - (eng_list[0] * exponent)) /
                               (1 - exponent))
    elif pref["method"] == "mix":
        c_num = pref["cardinal_num"]
        coef = [exp(c_num[0]**2 + c_num[1]**2 + 3 * c_num[2]),
                exp(3 * c_num[0] + c_num[1]**2 + c_num[2]**2),
                exp(c_num[0]**2 + 3 * c_num[1] + c_num[2]**2)]
        for eng_list in [hf_list, corr_list]:
            eng_lim.append(
                (coef[0] * (exp(c_num[0]) * eng_list[0]
                            - exp(c_num[1]) * eng_list[1])
                 + coef[1] * (exp(c_num[1]) * eng_list[1]
                              - exp(c_num[2]) * eng_list[2])
                 + coef[2] * (exp(c_num[2]) * eng_list[2]
                              - exp(c_num[0]) * eng_list[0]))
                / (coef[0] * (exp(c_num[0]) - exp(c_num[1]))
                   + coef[1] * (exp(c_num[1]) - exp(c_num[2]))
                   + coef[2] * (exp(c_num[2]) - exp(c_num[0]))))

    if pref["full_output"]:
        return result({"HF": eng_lim[0], "HF points": hf_list,
                       "Corr": eng_lim[1], "Corr points": corr_list,
                       "Total": eng_lim[0] + eng_lim[1]}, pref)
    else:
        return result(eng_lim[0] + eng_lim[1], pref)


def out2zpe(*args, **kwargs):
    """
    Returns:
        gaussian_zpe (float or None in Hartree)
    Args:
        input: gaussian output_file (str)
        pref: preference dict with above parameters as keys
    Note:
        This function is also applicable for MP2.

        out2zpe("MP2.out") = freq2zpe(out2freq("MP2.out")) + out2eng("MP2.out")
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref = update_pref(args, kwargs, pref)
    zpe = shell_cmd("grep -a 'zero-point' {} | tail -n 1 | awk '{{print $7}}'"
                    .format(escaped_file_name(pref["input"])))
    if zpe:
        zpe = float(zpe[-1])
    else:
        zpe = None
    return result(zpe, pref)


def out2freq(*args, **kwargs):
    """
    Returns:
        freq_list: wavenumber list in cm^-1 (list of float)
    Args:
        input: gaussian output_file (str)
        scaling_factor: to multiply to the frequencies (float)
        raise_read_error: True will raise Exception if detect no intensity
            (bool)
        bend_over_freq_range: list of lower and upper limits for bending
            overtone (list of float)
        symm_freq_range: list of lower and upper limits for symmetrical
            stretching (list of float)
        asymm_freq_range: list of lower and upper limits for asymmetrical
            stretching (list of float)
        J: for constructing anharmonic matric (float)
        pref: preference dict with above parameters as keys
    Note:
        In Gaussian output files, the Frequencies are acutally wavenumbers
        in cm^-1. The real frequency (Hz) = 2*pi*c(cm/s)*wavenumber(cm^-1)

        If frequencies are not existed in the output file and raise_read_error
        = False, it will return [].
        If atom_num cannot be detected, it will output full frequencies
        extracted from the output file.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    # pref["scaling_factor"] = 1.0
    pref["raise_read_error"] = False
    pref["bend_over_freq_range"] = None
    pref["symm_freq_range"] = None
    pref["asymm_freq_range"] = None
    pref["J"] = None
    pref = update_pref(args, kwargs, pref)
    freq_str_list = shell_cmd("grep -a 'Frequencies -- ' {}"
                              .format(escaped_file_name(pref["input"])))
    freq_list = []
    if freq_str_list:
        for freq_str in freq_str_list:
            freq_tmp_list = freq_str.split()
            for freq_tmp in freq_tmp_list[2:]:
                freq_list.append(float(freq_tmp))
                # freq_list.append(float(freq_tmp) * pref["scaling_factor"])
    elif pref["raise_read_error"]:
        raise Exception("Frequencies not found in {}".format(pref["input"]))
    if freq_list and \
            pref["bend_over_freq_range"] and \
            pref["symm_freq_range"] and \
            pref["asymm_freq_range"] and \
            pref["J"]:
        bend_over_freq = 0.0
        bend_over_freq_num = 0.0
        symm_freq = 0.0
        symm_freq_num = 0.0
        asymm_freq = 0.0
        asymm_freq_num = 0.0
        for freq in freq_list:
            if freq >= pref["bend_over_freq_range"][0] and \
                    freq <= pref["bend_over_freq_range"][-1]:
                bend_over_freq += freq
                bend_over_freq_num += 1.0
            if freq >= pref["symm_freq_range"][0] and \
                    freq <= pref["symm_freq_range"][-1]:
                symm_freq += freq
                symm_freq_num += 1.0
            if freq >= pref["asymm_freq_range"][0] and \
                    freq <= pref["asymm_freq_range"][-1]:
                asymm_freq += freq
                asymm_freq_num += 1.0
        if symm_freq_num == asymm_freq_num and \
                bend_over_freq_num > 0:
            bend_over_freq = bend_over_freq / bend_over_freq_num * 2
            symm_freq = symm_freq / symm_freq_num
            asymm_freq = asymm_freq / asymm_freq_num
            mat = [[bend_over_freq, pref["J"], 0],
                   [pref["J"], symm_freq, 0],
                   [0, 0, asymm_freq]]
            tmp_eigen_dict = eigen_dict(mat)
            if "eigenvalues" in tmp_eigen_dict:
                freq_list = [freq_list, list(tmp_eigen_dict["eigenvalues"])]
        else:
            print("\033[1;31mError! Inconsistent numbers\033[0m")
            print("bend_over_num=", bend_over_freq_num)
            print("symm_num=", symm_freq_num)
            print("asymm_num=", asymm_freq_num)
    return result(freq_list, pref)


def out2freq_legacy(*args, **kwargs):
    """
    Returns:
        freq_list: wavenumber list in cm^-1 (list of float)
    Args:
        input: gaussian output_file (str)
        raise_read_error: True will raise Exception if detect no intensity
            (bool)
        pref: preference dict with above parameters as keys
    Note:
        In Gaussian output files, the Frequencies are acutally wavenumbers
        in cm^-1. The real frequency (Hz) = 2*pi*c(cm/s)*wavenumber(cm^-1)

        If frequencies are not existed in the output file and raise_read_error
        = False, it will return [].
        If atom_num cannot be detected, it will output full frequencies
        extracted from the output file.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["raise_read_error"] = True
    pref = update_pref(args, kwargs, pref)
    escaped_file = escaped_file_name(pref["input"])
    freq_str_list = shell_cmd("grep -a 'Frequencies -- ' {}"
                              .format(escaped_file))
    freq_list = []
    if freq_str_list:
        for freq_str in freq_str_list:
            freq_tmp_list = freq_str.split()
            for freq_tmp in freq_tmp_list[2:]:
                freq_list.append(float(freq_tmp))
    elif pref["raise_read_error"]:
        raise Exception("Frequencies not found in {}".format(pref["input"]))
    atom_num_info = shell_cmd("grep -a 'NAtoms=' {} -m 1"
                              .format(escaped_file))
    if atom_num_info:
        atom_num = int(atom_num_info[0].split()[1])
        freq_list = freq_list[(-3 * atom_num + 6):]
    else:
        test_line = shell_cmd(
            "grep -a ' ---------------------------------------------------------------------$' {} -m 3 -n | awk -F':' '{{print $1}}'"
            .format(escaped_file))
        if test_line:
            first_line = int(test_line[0])
            second_line = int(test_line[1])
            third_line = int(test_line[2])
            if second_line - first_line == 3:
                atom_num = third_line - second_line - 1
                freq_list = freq_list[(-3 * atom_num + 6):]
        else:
            print("Warning! Cannot parse atom number")
            print("Will output full freq_list")
    return result(freq_list, pref)


def out2freq_dict(*args, **kwargs):
    """
    Returns:
        {"freq": [], "atom_id":[[]], "vib_vec":[[[]]], "vib_norm": [[]]} (dict)
    Args:
        input: gaussian output_file (str)
        norm_threshold: the minimum norm of vib. vectors to be counted in
            atom_id, vib_vec, and vib_norm (float)
        pref: preference dict with above parameters as keys
    Note:
        The atom_id, vib_vec, and vib_norm will be sorted by vib_norm from
        large norm to small norm.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["norm_threshold"] = 0.01
    pref = update_pref(args, kwargs, pref)
    freq_dict = {}
    norm_threshold_sq = pref["norm_threshold"] ** 2
    line1, line2, line3 = shell_cmd("grep -a ' ---------------------------------------------------------------------$' {} -m 3 -n | awk -F':' '{{print $1}}'"
                                    .format(escaped_file_name(pref["input"])))
    line1 = int(line1)
    line2 = int(line2)
    line3 = int(line3)
    if line2 - line1 - 1 == 2:
        atom_num = line3 - line2 - 1
    else:
        print("Cannot find atom num")
        return result(freq_dict, pref)
    freq_str_list = shell_cmd("grep -a -A {} 'Frequencies -- ' {}"
                              .format(atom_num + 5,
                                      escaped_file_name(pref["input"])))
    freq_list = []
    atom_id_list = []
    vib_vec_list = []
    vib_norm_list = []
    vec_line = -1
    for freq_str in freq_str_list:
        if "Frequencies --" in freq_str:
            freq_atom_id_list = []
            freq_vib_vec_list = []
            freq_vib_norm_list = []
            tmp_freq_list = freq_str.split()[2:]
            for tmp_freq in tmp_freq_list:
                freq_list.append(float(tmp_freq))
                freq_atom_id_list.append([])
                freq_vib_vec_list.append([])
                freq_vib_norm_list.append([])
            freq_num = len(tmp_freq_list)
        elif "Atom  AN" in freq_str:
            vec_line = atom_num
        elif vec_line > 0:
            tmp_vec = freq_str.split()
            atom_id = int(tmp_vec[0]) - 1  # starts from 0
            atom_type = number2atom[int(tmp_vec[1])]
            str_vib_vec = tmp_vec[2:]
            float_vib_vec = []
            for str_vec in str_vib_vec:
                float_vib_vec.append(float(str_vec))
            for freq_id in range(freq_num):
                vib_norm_sq = \
                    float_vib_vec[freq_id * 3] ** 2 + \
                    float_vib_vec[freq_id * 3 + 1] ** 2 + \
                    float_vib_vec[freq_id * 3 + 2] ** 2
                if vib_norm_sq >= norm_threshold_sq:
                    freq_atom_id_list[freq_id].append(atom_id)
                    freq_vib_vec_list[freq_id]\
                        .append([atom_type,
                                 float_vib_vec[freq_id * 3],
                                 float_vib_vec[freq_id * 3 + 1],
                                 float_vib_vec[freq_id * 3 + 2]])
                    freq_vib_norm_list[freq_id].append(sqrt(vib_norm_sq))
            vec_line -= 1
        elif vec_line == 0:
            for I0, freq_vib_norm in enumerate(freq_vib_norm_list):
                norm_sort_index = sort_index(freq_vib_norm, order="descent")
                freq_atom_id_list[I0] = sort_with_index(freq_atom_id_list[I0],
                                                        norm_sort_index)
                freq_vib_vec_list[I0] = sort_with_index(freq_vib_vec_list[I0],
                                                        norm_sort_index)
                freq_vib_norm_list[I0] = \
                    sort_with_index(freq_vib_norm_list[I0],
                                    norm_sort_index)
            atom_id_list += freq_atom_id_list
            vib_vec_list += freq_vib_vec_list
            vib_norm_list += freq_vib_norm_list
            vec_line -= 1
    freq_dict["freq"] = freq_list
    freq_dict["atom_id"] = atom_id_list
    freq_dict["vib_vec"] = vib_vec_list
    freq_dict["vib_norm"] = vib_norm_list
    return result(freq_dict, pref)


def out2freq_vec(*args, **kwargs):
    """
    Returns:
        vec_list: (list of array)
        full_vec_dict: (dict, if full_output is True)
    Args:
        input: gaussian output_file (str)
        full_output: will return all the corresponing frequency, elements and
            xyz coordinates in standard orientation
        raise_read_error: True will raise Exception if detect no intensity
            (bool)
        pref: preference dict with above parameters as keys
    Note:
        Will only work if hpmodes is used in gaussian calculation\n\n
        Each set of list of force vector corresponds to the frequency in
        out2freq and the list of vector corresponds to the xyz of atoms in
        standard orientation from g09_out2xyz in the same order
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["full_output"] = False
    pref["raise_read_error"] = True
    pref = update_pref(args, kwargs, pref)

    atom_num_info = shell_cmd("grep -a 'NAtoms=' {} -m 1".format(pref["input"]))
    atom_num = int(atom_num_info[0].split()[1])
    atom_num_xyz = atom_num * 3
    vec_str_list = shell_cmd("awk '/Coord Atom Element:/ {{for(i=1; i<={}; i++) {{getline; print}}}}' {} | awk '{{for (i=4; i<NF; i++) printf $i \" \"; print $NF}}'"
                             .format(atom_num_xyz, pref["input"]))
    vec_list = []
    if vec_str_list:
        tmp_vec_list = []
        for i, vec_str in enumerate(vec_str_list):
            if i < atom_num_xyz:
                tmp_vec_list.append(list(map(float, vec_str.split())))
            else:
                tmp_vec_list[i % atom_num_xyz] += \
                    list(map(float, vec_str.split()))
        for arr in array(tmp_vec_list).T:
            vec_list.append(arr.reshape(atom_num, 3))
    elif pref["raise_read_error"]:
        raise Exception("Frequencies not found in {}".format(pref["input"]))

    if pref["full_output"]:
        xyz = g09_out2xyz(pref["input"], "Standard")
        atom_name = array([x[0] for x in xyz], dtype=object)
        freq = out2freq(pref["input"])
        mass = {}
        for an in atom_name:
            mass[an] = atomic_mass[an]
        tmp_vec_dict = {}
        for i, f in enumerate(freq):
            tmp_vec_dict[f] = hstack((atom_name.reshape(atom_name.size, 1),
                                      vec_list[i])).tolist()
        full_vec_dict = {"xyz": xyz,
                         "mass": mass,
                         "freq": freq,
                         "freq_vec": tmp_vec_dict}

        return result(full_vec_dict, pref)
    else:
        return result(vec_list, pref)


def out2freq_red_mass(*args, **kwargs):
    """
    Returns:
        red_mass: (list of float)
    Args:
        input: gaussian output file (str)
        pref: preference dict with above parameters as keys
    Note:
        Reduced mass from gaussian as defined in https://gaussian.com/vib/ which
        is the invered sum of cartesian displacement squared (Eq. 13)
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref = update_pref(args, kwargs, pref)

    mass_str_list = shell_cmd("grep -a 'Red. masses -- ' {}"
                              .format(pref["input"]))
    mass_list = []
    if mass_str_list:
        for mass_str in mass_str_list:
            mass_tmp_list = mass_str.split()
            for mass_tmp in mass_tmp_list[3:]:
                mass_list.append(float(mass_tmp))

    return result(mass_list, pref)


def out2dipole(*args, **kwargs):
    """
    Returns:
        list of dipole [dipole_x, dipole_y, dipole_z]: (list)
    Args:
        input: gaussian output file (str)
        pref: preference dict with above parameters as keys
    Note:
        Unit in Debye:
            1 Debye = 0.2081943 e A
            1 e A = 4.80320 Debye
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref = update_pref(args, kwargs, pref)
    dipole_list = shell_cmd("grep -A1 'Dipole moment' {} | tail -n 1"
                            .format(pref["input"]))[0].split()
    dipole = [float(dipole_list[1]),
              float(dipole_list[3]),
              float(dipole_list[5])]
    return result(dipole, pref)


def out2dipole_der(*args, **kwargs):
    """
    Returns:
        list of dipole_derivatives: (list of list of floats)
    Args:
        input: gaussian output file (str)
        key: search key for dipole derivatives (str)
        pref: preference dict with above parameters as keys
    Note:
        To print out the dipole derivatives in Gaussian, use freq without opt
        and add the iop(7/33=1) option. The IOP settings can not propagate from
        opt to freq.

        Opt should not be used.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["dipole_der_key"] = "Dipole derivative wrt mode"
    pref = update_pref(args, kwargs, pref)
    tmp_shell_list = shell_cmd("grep -a '{}' {}"
                               .format(pref["dipole_der_key"],
                                       escaped_file_name(pref["input"])))

    dipole_der_list = []
    if tmp_shell_list:
        for tmp_str in tmp_shell_list:
            tmp_list = tmp_str.split()
            dipole_der_list.append([float(tmp_list[5].replace("D", "E")),
                                    float(tmp_list[6].replace("D", "E")),
                                    float(tmp_list[7].replace("D", "E"))])
    else:
        print("Can not find the dipole derivatives with the keyword: \n{}"
              .format(pref["dipole_der_key"]))
    return result(dipole_der_list, pref)


def dipole_der2inten(*args, **kwargs):
    """
    Returns:
        list of intensities: (list of float)
    Args:
        input: list of dipole derivatives from iOp(7/33=1)
        pref: preference dict with above parameters as keys
    Note:
        Gaussian has converted the unit of the dipole derivatives into
        sqrt(km/mol)

        The unit of the IR intensity is km/mol.
        See: https://mattermodeling.stackexchange.com/questions/5021/what-units-are-used-in-gaussian-16-for-dipole-derivatives-output
    """
    pref = OrderedDict()
    pref["input"] = []
    pref = update_pref(args, kwargs, pref)
    inten_list = []
    for mode in pref["input"]:
        inten_list.append(mode[0] ** 2 + mode[1] ** 2 + mode[2] ** 2)
    return result(inten_list, pref)


def out2inten(*args, **kwargs):
    """
    Returns:
        inten_list (list of float)
    Args:
        input: gaussian output_file (str)
        raise_read_error: True will raise Exception if detect no intensity
            (bool)
        pref: preference dict with above parameters as keys
    Note:
        If intensities are not existed in the output file and raise_read_error
        = False, it will return [].
        If atom_num cannot be detected, it will output full intensities
        extracted from the output file.

        The unit of the intensity in Gaussian is km/mol.

        Intensities of Gaussian are computed by the sum of the square of the
        dipole derivatives in x, y, z directions.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["raise_read_error"] = False
    pref = update_pref(args, kwargs, pref)
    inten_str_list = shell_cmd("grep -a 'IR Inten    -- ' {}"
                               .format(escaped_file_name(pref["input"])))
    inten_list = []
    if inten_str_list:
        for inten_str in inten_str_list:
            inten_tmp_list = inten_str.split()
            for inten_tmp in inten_tmp_list[3:]:
                inten_list.append(float(inten_tmp))
    elif pref["raise_read_error"]:
        raise Exception("Intensities not found in {}".format(pref["input"]))
    return result(inten_list, pref)


def out2inten_legacy(*args, **kwargs):
    """
    Returns:
        inten_list (list of float)
    Args:
        input: gaussian output_file (str)
        raise_read_error: True will raise Exception if detect no intensity
            (bool)
        pref: preference dict with above parameters as keys
    Note:
        If intensities are not existed in the output file and raise_read_error
        = False, it will return [].
        If atom_num cannot be detected, it will output full intensities
        extracted from the output file.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["raise_read_error"] = False
    pref = update_pref(args, kwargs, pref)
    escaped_file = escaped_file_name(pref["input"])
    inten_str_list = shell_cmd("grep -a 'IR Inten    -- ' {}"
                               .format(escaped_file))
    inten_list = []
    if inten_str_list:
        for inten_str in inten_str_list:
            inten_tmp_list = inten_str.split()
            for inten_tmp in inten_tmp_list[3:]:
                inten_list.append(float(inten_tmp))
    elif pref["raise_read_error"]:
        raise Exception("Intensities not found in {}".format(pref["input"]))
    atom_num_info = shell_cmd("grep -a 'NAtoms=' {} -m 1"
                              .format(escaped_file))
    if atom_num_info:
        atom_num = int(atom_num_info[0].split()[1])
        inten_list = inten_list[(-3 * atom_num + 6):]
    else:
        test_line = shell_cmd(
            "grep -a ' ---------------------------------------------------------------------$' {} -m 3 -n | awk -F':' '{{print $1}}'"
            .format(escaped_file))
        if test_line:
            first_line = int(test_line[0])
            second_line = int(test_line[1])
            third_line = int(test_line[2])
            if second_line - first_line == 3:
                atom_num = third_line - second_line - 1
                inten_list = inten_list[(-3 * atom_num + 6):]
        else:
            print("Error! Cannot parse atom number")
            print("Will output full inten_list")
    return result(inten_list, pref)


def out2anharm_ir(*args, **kwargs):
    """
    Returns:
        ir_dict: {"freq_list": [], "inten_list": []}
    Args:
        input: gaussian output file or ir_dict (str or dict)
        pref: preference dict with above parameters as keys
    Note:
        Source code was written by Dr. Cheng-Chau Chiu.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref = update_pref(args, kwargs, pref)
    section = {
        "Fundamental Bands": 0,
        "Overtones": 0,
        "Combination Bands": 0}
    section_E = {
        "Fundamental Bands": 2,
        "Overtones": 2,
        "Combination Bands": 3}
    section_I = {
        "Fundamental Bands": 4,
        "Overtones": 3,
        "Combination Bands": 4}
    freq_list = []
    inten_list = []
    ir_dict = {}
    with open(pref["input"], "r") as file_obj:
        ignore = True
        for line in file_obj:
            if "Anharmonic Infrared Spectroscopy" in line:
                ignore = False
            sp = line.split()
            if not ignore:
                band_type = line.strip()
                if band_type in section.keys():
                    section[band_type] += 1
                    if section[band_type] == 2:
                        # Break condition for G16
                        ir_dict["freq_list"] = freq_list
                        ir_dict["inten_list"] = inten_list
                        break
                    e_n = section_E[band_type]
                    i_n = section_I[band_type]
                elif len(sp) > 1:
                    if "(" in sp[0] and ")" == sp[0][-1] and \
                            "Mode" not in sp[0]:
                        freq_list.append(float(sp[e_n]))
                        inten_list.append(float(sp[i_n]))
                if "GradGradGradGradGradGradGradGrad" in line:
                    # Break condition for G09
                    ir_dict["freq_list"] = freq_list
                    ir_dict["inten_list"] = inten_list
                    break
    return result(ir_dict, pref)


def out2ir(*args, **kwargs):
    """
    Returns:
        ir_dict: {"freq_list": [], "inten_list": []}
    Args:
        input: gaussian output file or ir_dict (str or dict)
        weighted_list: if not None, it will re-weight the input
            intensities (list of float or None)
        scan_freq_list: the complete frequency range (list of float)
        alpha: for power-law function (float)
        beta: for power-law function (float)
        free_oh_freq: for power-law function (float)
        fix_gamma_list: instead of power-law function (list of float)
        fix_gamma_freq_list: freq range for fix_gamma_list (list of list)
        fix_gamma_min_inten_list: minimum intensities for fix_gamma_list
            (list of float)
        pref: preference dict with above parameters as keys
    Note:
        gamma is full width at half maximum (FWHM).\n\n

        All frequency units are cm^{-1}.\n\n

        fix_gamma_freq_list, fix_gamma_list, fix_gamma_min_inten_list must
        have the same dimension.\n\n

        The inner range list of fix_gamma_freq_list should not have any
        operlapping.\n\n

        If fix_gamma_min_inten = -1, it will accept all the intensities.

        For larger gamma in fix_gamma_list, the frequency band will become
        broader and lower. For smaller gamma, the band will become sharper and
        higher.
    Examples:
        out2ir("0.out")
        ir_dict = {"freq_list": [1,2,...], "inten_list": [0, 12, ...]}
        out2ir(ir_dict)
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["weighted_list"] = []
    pref["scan_freq_list"] = []
    pref["alpha"] = 0.0009
    pref["beta"] = 1.9
    pref["free_oh_freq"] = 3678
    pref["fix_gamma_list"] = [5, 20]
    pref["fix_gamma_freq_list"] = [[2750, 3250], [3600, 4000]]
    pref["fix_gamma_min_inten_list"] = [100, -1]
    pref = update_pref(args, kwargs, pref)
    if isinstance(pref["input"], dict):
        # Detect input ir_dict
        freq_input_list = pref["input"]["freq_list"]
        inten_input_list = pref["input"]["inten_list"]
    else:
        freq_input_list = out2freq(pref["input"])
        inten_input_list = out2inten(pref["input"])
    if pref["weighted_list"]:
        weighted_inten_input_list = []
        for I0, inten in enumerate(inten_input_list):
            weighted_inten_input_list.append(inten * pref["weighted_list"][I0])
    else:
        weighted_inten_input_list = inten_input_list

    inten_out_list = [0.0] * len(pref["scan_freq_list"])
    for I0, freq_out in enumerate(pref["scan_freq_list"]):
        for I1, freq in enumerate(freq_input_list):
            use_power_law = True
            for I2, fix_gamma_freq in enumerate(pref["fix_gamma_freq_list"]):
                if freq >= fix_gamma_freq[0] and freq <= fix_gamma_freq[1]:
                    if pref["fix_gamma_min_inten_list"][I2] < 0 or \
                            inten_input_list[I1] <= \
                            pref["fix_gamma_min_inten_list"][I2]:
                        # print("CH freq:", freq, inten_input_list[I1], pref["fix_gamma_min_inten_list"][I2])
                        gamma = pref["fix_gamma_list"][I2]
                        use_power_law = False
            if use_power_law:  # Use abs() to prevent imaginary gamma
                gamma = pref["alpha"] * (abs(pref["free_oh_freq"]
                                         - freq)) ** pref["beta"]
            denominator = (freq_out - freq) ** 2 + (gamma / 2.0) ** 2
            inten_out_list[I0] += (1.0 / pi) * (gamma / denominator * 0.5) \
                * weighted_inten_input_list[I1]
    return result({"freq_list": pref["scan_freq_list"],
                   "inten_list": inten_out_list}, pref)


def out2ir_no_reweight(*args, **kwargs):
    """
    Returns:
        ir_dict: {"freq_list": [], "inten_list": []}
    Args:
        input: gaussian output file or ir_dict (str or dict)
        scan_freq_list: the complete frequency range (list of float)
        alpha: for power-law function (float)
        beta: for power-law function (float)
        free_oh_freq: for power-law function (float)
        fix_gamma_list: instead of power-law function (list of float)
        fix_gamma_freq_list: freq range for fix_gamma_list (list of list)
        fix_gamma_min_inten_list: minimum intensities for fix_gamma_list
            (list of float)
        pref: preference dict with above parameters as keys
    Note:
        This version will not re-weight any intensity.\n
        All frequency units are cm^{-1}.\n\n

        fix_gamma_freq_list, fix_gamma_list, fix_gamma_min_inten_list must
        have the same dimension.\n\n

        The inner range list of fix_gamma_freq_list should not have any
        operlapping.\n\n

        If fix_gamma_min_inten = -1, it will accept all the intensities.

        Warning! Using np.array will become very slow.
    Examples:
        out2ir("0.out")
        ir_dict = {"freq_list": [1,2,...], "inten_list": [0, 12, ...]}
        out2ir(ir_dict)
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["scan_freq_list"] = []
    pref["alpha"] = 0.0009
    pref["beta"] = 1.9
    pref["free_oh_freq"] = 3678
    pref["fix_gamma_list"] = [5, 20]
    pref["fix_gamma_freq_list"] = [[2750, 3250], [3600, 4000]]
    pref["fix_gamma_min_inten_list"] = [100, -1]
    pref = update_pref(args, kwargs, pref)
    if isinstance(pref["input"], dict):
        # Detect input ir_dict
        freq_input_list = pref["input"]["freq_list"]
        inten_input_list = pref["input"]["inten_list"]
    else:
        freq_input_list = out2freq(pref["input"])
        inten_input_list = out2inten(pref["input"])
    inten_out_list = [0.0] * len(pref["scan_freq_list"])
    for I0, freq_out in enumerate(pref["scan_freq_list"]):
        for I1, freq in enumerate(freq_input_list):
            use_power_law = True
            for I2, fix_gamma_freq in enumerate(pref["fix_gamma_freq_list"]):
                if freq >= fix_gamma_freq[0] and freq <= fix_gamma_freq[1]:
                    if pref["fix_gamma_min_inten_list"][I2] < 0 or \
                            inten_input_list[I1] <= \
                            pref["fix_gamma_min_inten_list"][I2]:
                        # print("CH freq:", freq, inten_input_list[I1], pref["fix_gamma_min_inten_list"][I2])
                        gamma = pref["fix_gamma_list"][I2]
                        use_power_law = False
            if use_power_law:  # Use abs() to prevent imaginary Gamma
                gamma = pref["alpha"] * (abs(pref["free_oh_freq"]
                                         - freq)) ** pref["beta"]
            denominator = (freq_out - freq) ** 2 + (gamma / 2.0) ** 2
            inten_out_list[I0] += (1.0 / pi) * (gamma / denominator * 0.5) \
                * inten_input_list[I1]
    return result({"freq_list": pref["scan_freq_list"],
                   "inten_list": inten_out_list}, pref)


def header_str(*args, **kwargs):
    """
    Returns:
        header string: (str)
    Args:
        header: template name (str)
        basis: basis set name (str)
        pref: preference dict with above parameters as keys
    Note:

    """
    pref = OrderedDict()
    pref["header"] = "B3LYP_OPT"
    pref["basis"] = "6-31+G(d)"
    pref = update_pref(args, kwargs, pref)
    tmp_header_str = ""
    if "#" in pref["header"].lower():
        tmp_header_str = pref["header"]
    elif isfile(pref["header"]):
        escaped_file = escaped_file_name(pref["header"])
        header_type = xyz_type(escaped_file)
        if header_type == "g09_out":
            tmp_header_str = out2header(escaped_file, header_type="g09_inp")
        elif header_type == "g09_inp":
            tmp_header_str = inp2header(escaped_file)
    return result(tmp_header_str, pref)


def inp2header(*args, **kwargs):
    """
    Returns:
        header_str: nproc, mem, chk, keywords, charge and multiplicity (str)
    Args:
        input: gaussian input file (str)
        header_file: if not None will write the header to a header file (str)
        pref: preference dict with above parameters as keys
    Note:
        Only accept input file format
    """
    pref = OrderedDict()
    pref["input"] = "0.inp"
    pref["header_file"] = None
    pref = update_pref(args, kwargs, pref)
    escaped_file = escaped_file_name(pref["input"])
    mach = "\n".join(shell_cmd("grep -a '%' {}"
                               .format(escaped_file)))
    mach = mach.replace(" ", "")
    if not mach:
        mach = "%NprocShared=6\n%mem=8Gb\n"
    keywords = shell_cmd("grep -a -A 6 '#' {}".format(escaped_file))
    option = keywords[0]
    line_num = len(keywords)
    chrg_multi = "0 1"
    charge_line = -1
    for I0 in range(line_num):
        inv_line_id = line_num - I0 - 1
        try:
            keyword_list = keywords[inv_line_id].split()
            if keyword_list:
                find_molecule_spec = True
                for keyword in keyword_list:
                    delta_keyword = float(keyword) - float(int(keyword))
                    if delta_keyword != 0:
                        find_molecule_spec = False
                if find_molecule_spec:
                    chrg_multi = keywords[inv_line_id]
                    charge_line = inv_line_id
                    break
        except:
            continue
    remark = "".join(keywords[1:charge_line])
    tmp_header_str = ""
    mach = mach.strip()
    for mach_str in mach.split("\n"):
        tmp_header_str += mach_str.strip() + "\n"
    tmp_header_str += \
        option.strip() + "\n\n" + remark.strip() + "\n\n" + chrg_multi.strip() + "\n"
    if pref["header_file"]:
        with open(pref["header_file"], "w") as header_obj:
            header_obj.write(tmp_header_str)
        print("Write header to \033[1;34m{}\033[0m"
              .format(pref["header_file"]))
    return result(tmp_header_str, pref)


def out2header(*args, **kwargs):
    """
    Returns:
        header_str: nproc, mem, chk, keywords, charge and multiplicity (str)
    Args:
        input: gaussian output file (str)
        header_type: header_str style: "g09_inp" or "g09_out" (str)
        header_file: if not None will write the header to a header file (str)
        pref: preference dict with above parameters as keys
    Note:
        header_type = g09_out will generate the style of the Gaussian output
        file. This is useful for resizing the Gaussian output file.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["header_type"] = "g09_out"
    pref["header_file"] = None
    pref = update_pref(args, kwargs, pref)
    escaped_file = escaped_file_name(pref["input"])
    mach = "\n".join(shell_cmd("grep -a '%' {}"
                               .format(escaped_file)))
    mach = mach.replace(" ", "")
    keywords = shell_cmd("grep -a -m 1 -A 2 '#' {}".format(escaped_file))
    if not keywords:
        return result("", pref)
    if "---------------" not in keywords[1] and \
            "--------------" not in keywords[2] and keywords[1] and keywords[2]:
        option = keywords[0].strip() + keywords[1][1:].rstrip() + \
            keywords[2][1:].rstrip()
    elif "---------------" not in keywords[1] and keywords[1]:
        option = keywords[0].strip() + keywords[1][1:].rstrip()
    else:
        option = keywords[0]
    remark_charge_multiplicity = \
        shell_cmd("grep -a -C 3 'Multiplicity' {}".format(escaped_file))
    remark = remark_charge_multiplicity[0]
    tmp_charge_multiplicity = remark_charge_multiplicity[3]
    if "supermolecule" in tmp_charge_multiplicity:
        if pref["header_type"] == "g09_inp":
            charge_multiplicity = ""
            for fragment_str in remark_charge_multiplicity[4:]:
                if "in fragment" in fragment_str:
                    fragment_str_list = fragment_str.split()
                    charge_multiplicity += "{} {} ".format(fragment_str_list[2],
                                                           fragment_str_list[5])
            charge_multiplicity = charge_multiplicity.strip() + "\n"
        elif pref["header_type"] == "g09_out":
            charge_multiplicity = tmp_charge_multiplicity.strip() + "\n"
            for fragment_str in remark_charge_multiplicity[4:]:
                if "in fragment" in fragment_str:
                    charge_multiplicity += fragment_str.strip() + "\n"
                else:
                    break
    elif "=" in tmp_charge_multiplicity:
        tmp_list = tmp_charge_multiplicity.split()
        if pref["header_type"] == "g09_inp":
            charge_multiplicity = "{} {}".format(tmp_list[2],
                                                 tmp_list[5])
        elif pref["header_type"] == "g09_out":
            charge_multiplicity = "Charge = {} Multiplicity = {}".format(tmp_list[2],
                                                                         tmp_list[5])
    tmp_header_str = ""
    mach = mach.strip()
    for mach_str in mach.split("\n"):
        tmp_header_str += mach_str.strip(" ") + "\n"
    tmp_header_str += \
        option.strip() + "\n\n" + remark.strip() + "\n\n" + charge_multiplicity.strip() + "\n"
    if pref["header_file"]:
        with open(pref["header_file"], "w") as header_obj:
            header_obj.write(tmp_header_str)
        print("Write header to  \033[1;34m{}\033[0m".format(pref["header_file"]))
    return result(tmp_header_str, pref)


def out2bsse(*args, **kwargs):
    """
    Returns:
        gaussian_bsse_eng: (float or 0 for None in Hartree)
    Args:
        input: gaussian output_file (str)
        pref: preference which includes all parameters (dict)
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref = update_pref(args, kwargs, pref)
    bsse_eng = \
        shell_cmd("grep -a 'BSSE energy =' {} | awk '{{print $4}}'"
                  .format(escaped_file_name(pref["input"])))
    if bsse_eng:
        bsse_eng = float(bsse_eng[-1])
    else:
        print("\033[93mNo BSSE calculation\033[0m")
        bsse_eng = 0
    return result(bsse_eng, pref)


def out2vol(*args, **kwargs):
    """
    Returns:
        gaussian_vol: (float or -1 for None)
    Args:
        input: gaussian output_file (str)
        molar: output molar volume or volume per molecule (bool, default=True)
        pref: preference which includes all parameters (dict)
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["molar"] = True
    pref = update_pref(args, kwargs, pref)
    molar_vol = shell_cmd(
        "grep -a 'Molar volume =' {} | head -n 1 | cut -d '(' -f2 | cut -d 'c' -f1"
        .format(escaped_file_name(pref["input"])))

    if molar_vol:
        molar_vol = float(molar_vol[-1])
        if pref["molar"]:
            molar_vol /= Avogadro["mol^-1"]
    else:
        molar_vol = None
    return result(molar_vol, pref)


def out2rot_const(*args, **kwargs):
    """
    Returns:
        rot_const_list: in GHZ (list of float)
    Args:
        input: gaussian output_file (str)
        pref: preference dict with above parameters as keys
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref = update_pref(args, kwargs, pref)
    rot_const_info = shell_cmd(
        "grep -a 'Rotational constants' {} | tail -n 1 | awk '{{print $4\" \"$5\" \"$6}}'"
        .format(escaped_file_name(pref["input"])))
    if rot_const_info:
        rot_const_list = rot_const_info[-1].split()
        if len(rot_const_list) == 3:
            return result([float(rot_const_list[0]),
                           float(rot_const_list[1]),
                           float(rot_const_list[2])], pref)
        else:
            return([], pref)
    else:
        return result([], pref)


def check_out(*args, **kwargs):
    """
    Returns:
        0: if output file is OK! (int)
        -10: found abnormal terminated (int)
        -11: found small interatomic distance (int)
        -12: found galloc error (int)
        -20: found imaginary freq (int)
        -30: found no coordinate (int)
    Args:
        input: gaussian output file (str)
        check_normal: True will check if output file is normally terminated
            (bool)
        check_img_freq: True will check if output file contains imaginary freqs
            (bool)
        check_xyz: True will check if output file has no coordiate (bool)
        resume_last_step: True will rename the fail output file and rebuild
            the input file from the last step (bool).\n\n
    Note:
        For -11, -12, -30, and resume_last_step == True, it will not rebuild
        the input file. It will just rename the output file to *.out.fail.

        If the input file was exist, before rebuilding, it will be rename to
        *.inp.fail.

        pref: preference dict with above parameters as keys
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["check_normal"] = True
    pref["check_img_freq"] = True
    pref["check_xyz"] = True
    pref["resume_last_step"] = True
    pref = update_pref(args, kwargs, pref)
    exit_code = 0
    escaped_file = escaped_file_name(pref["input"])
    if pref["check_xyz"]:
        input_count = shell_cmd("grep -a 'Input orientation:' {} | wc -l"
                                .format(escaped_file))[0].strip()
        standard_count = shell_cmd("grep -a 'Standard orientation:' {} | wc -l"
                                   .format(escaped_file))[0].strip()
        if int(input_count) == 0 and int(standard_count) == 0:
            print("Error! No coordinates in {}"
                  .format(escaped_file))
            if pref["resume_last_step"]:
                print("Rename \033[1;34m{} \033[0mto \033[1;34m{}\033[1;31m.fail\033[0m"
                      .format(escaped_file, escaped_file))
                shell_cmd("mv -f {} {}.fail".format(escaped_file,
                                                    escaped_file))
            return result(-30, pref)
    if pref["check_normal"]:
        galloc_info = shell_cmd(
            "grep -a 'galloc:  could not allocate memory' {}"
            .format(escaped_file))
        normal_info = shell_cmd("grep -a 'Normal termination' {}"
                                .format(escaped_file))
        error_info = shell_cmd("grep -a 'Erro' {} | grep -a -v 'RMS Error'"
                               .format(escaped_file))
        small_distance_info = \
            shell_cmd("grep -a 'Small interatomic distances encountered:' {}"
                      .format(escaped_file))
        if galloc_info:
            print("\033[1;31mError! Could not allocate memory in \033[1;34m{}\033[0m"
                  .format(pref["input"]))
            if pref["resume_last_step"]:
                input_file = "{}.inp".format(escaped_file.split(".")[0])
                if isfile(input_file):
                    print("Rename \033[1;34m{} \033[0mto \033[1;34m{}\033[1;31m.fail\033[0m"
                          .format(input_file, input_file))
                    shell_cmd("mv -f {} {}.fail".format(input_file,
                                                        input_file))
                out2inp(escaped_file, output="input")
                print("Rename \033[1;34m{} \033[0mto \033[0m{}\033[1;31m.fail\033[0m"
                      .format(escaped_file, escaped_file))
                shell_cmd("mv -f {} {}.fail".format(escaped_file,
                                                    escaped_file))
            return result(-12, pref)
        if small_distance_info:
            print("\033[1;31mError! Small interatomic distances in \033[1;34m{}\033[0m"
                  .format(pref["input"]))
            if pref["resume_last_step"]:
                print("Rename \033[1;34m{} \033[0mto \033[1;34m{}\033[1;31m.fail\033[0m"
                      .format(escaped_file, escaped_file))
                shell_cmd("mv -f {} {}.fail".format(escaped_file,
                                                    escaped_file))
            return result(-11, pref)
        if not normal_info or error_info:
            print("\033[1;31mError! No normal termination in \033[1;34m{}\033[0m"
                  .format(pref["input"]))
            if pref["resume_last_step"]:
                input_file = "{}.inp".format(escaped_file.split(".")[0])
                if isfile(input_file):
                    print("Rename \033[1;34m{} \033[0mto \033[1;34m{}\033[1;31m.fail\033[0m"
                          .format(input_file, input_file))
                    shell_cmd("mv -f {} {}.fail".format(input_file,
                                                        input_file))
                out2inp(escaped_file, output="input")
                print("Rename \033[1;34m{} \033[0mto \033[1;34m{}\033[1;31m.fail\033[0m"
                      .format(escaped_file, escaped_file))
                shell_cmd("mv -f {} {}.fail".format(escaped_file,
                                                    escaped_file))
            return result(-10, pref)
    if pref["check_img_freq"]:
        freq_info_list = \
            shell_cmd("""grep -a "Frequencies -- " {} | awk -F"--" '{{print $NF}}'"""
                      .format(escaped_file))
        for idx, freq_info in enumerate(freq_info_list):
            for freq in freq_info.split():
                if float(freq) <= 0.0:
                    print("\033[1;31mError! Found imaginary/zero frequencies \033[1;36m{}\033[1;31m for \033[1;34m{}\033[0m"
                          .format(freq, pref["input"]))
                    if pref["resume_last_step"]:
                        input_file = "{}.inp".format(escaped_file.split(".")[0])
                        if isfile(input_file):
                            print("Rename \033[1;34m{} \033[0mto \033[1;34m{}\033[1;31m.fail\033[0m"
                                  .format(input_file, input_file))
                            shell_cmd("mv -f {} {}.fail".format(input_file,
                                                                input_file))
                        out2inp(escaped_file, output=input_file)
                        print("Rename \033[1;34m{} \033[0mto \033[1;34m{}\033[1;31m.fail\033[0m"
                              .format(escaped_file, escaped_file))
                        shell_cmd("mv -f {} {}.fail".format(escaped_file,
                                                            escaped_file))
                    return result(-20, pref)
    return result(exit_code, pref)


def out2inp(*args, **kwargs):
    """
    Returns:
        0: if succeeded
    Args:
        input: Gaussian output file (str)
        output: Gaussian input file or "input" (str)
        pref: preference dict with above parameters as keys
    Note:
        Rebuild the Gaussian input file from the last step of the output file.

        If output == "input", it will use the base name of input with ".inp"
        file extension.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["output"] = "input"
    pref = update_pref(args, kwargs, pref)
    escaped_file = escaped_file_name(pref["input"])
    if isfile(escaped_file):
        if pref["output"] == "input":
            inp_file = "{}.inp".format(escaped_file.split(".")[0])
        else:
            inp_file = escaped_file_name(pref["output"])
        xyz_list = xyz2list(escaped_file, frame_list=[-1])
        tmp_header_str = out2header(escaped_file, header_type="g09_inp")
        print("Rebuild {} from {}".format(inp_file, escaped_file))
        xyz2inp(xyz_list, output=inp_file, header=tmp_header_str)
    return result(0, pref)


def write_inp_file(*args, **kwargs):
    """
    Returns:
        0: if succeeded
    Args:
        output: Gaussian input file (str)
        input : xyz_file_name (str), list of list, or list of str
        header_file: header_file (str)
        header_str: header_string (str)
        use_chk_file: True will use output filename as chk_file
        ext_list: restrict the output extension name
        append_info: additional information after Xyz coordinate (str)
        pref: preference includes all parameters (dict)
    Note:
        If header_file exists, it will overwrite header_str.\n\n

        chk_file="output" will use the output name as chk_file.\n\n

        ext_list is for preventing overwriting the xyz file.\n\n

        If input = None, it will not store the xyz coordinates.\n\n

        This is the enhanced version of xyz_tools.write_inp_file()
    """
    pref = OrderedDict()
    pref["output"] = "output.inp"
    pref["input"] = [
        "H 0.075853 0.002082 -0.003417",
        "O -0.625132 1.192093 -0.438473",
        "H -0.651382 1.264230 -1.415388",
    ]
    pref["header_file"] = ""
    pref["header_str"] = "%NprocShared=12\n\
%mem=8Gb\n\
# b3lyp/6-31+G* opt=(maxcycles=100)\n\
\n\
remark here\n\
\n\
0 1\n"
    pref["use_chk_file"] = False
    pref["ext_list"] = ["inp", "gjf", "com"]
    pref["append_info"] = None
    pref = update_pref(args, kwargs, pref)
    print(pref["append_info"])
    if pref["output"].split(".")[-1] not in pref["ext_list"]:    # Safty check
        print("Something wrong....")
        print("output file: {}".format(pref["output"]))
        print("xyz_list or xyz_file: {}".format(pref["input"]))
        print("Maybe you use the source xyz_file (xyz_list)")
        print("as output_file")
        print("Please check the order of the argument or ext_list")
        print("Then try again")
        return result(1, pref)
    if not pref["input"]:    # not store xyz coordinate
        with open(pref["output"], "w") as inp_obj:
            if pref["use_chk_file"]:
                inp_obj.write("%chk={}.chk\n".
                              format(basename(pref["output"]).split(".")[0]))

            if pref["header_file"]:
                with open(pref["header_file"], "r") as header_obj:
                    for read_line in header_obj:
                        inp_obj.write(read_line)
            else:
                inp_obj.write(pref["header_str"])
            inp_obj.write("\n")
            if pref["append_info"]:
                inp_obj.write(pref["append_info"])

    elif isinstance(pref["input"], str):    # xyz file
        with Xyz(pref["input"], "r") as xyz_obj:
            if xyz_obj.frame_num == 1:
                xyz_str_list = xyz_obj.next_str()
                with open(pref["output"], "w") as inp_obj:
                    if pref["use_chk_file"]:
                        inp_obj.write("%chk={}.chk\n"
                                      .format(basename(pref["output"])
                                              .split(".")[0]))
                    if pref["header_file"]:
                        with open(pref["header_file"], "r") as header_obj:
                            for read_line in header_obj:
                                inp_obj.write(read_line)
                    else:
                        inp_obj.write(pref["header_str"])
                    for xyz in xyz_str_list:
                        inp_obj.write(xyz)
                    inp_obj.write("\n")
                    if pref["append_info"]:
                        inp_obj.write(pref["append_info"])
            else:
                for frame_id in range(xyz_obj.frame_num):
                    output_file = \
                        "{}_{}.xyz".format(pref["output"].split(".")[0],
                                           frame_id)
                    xyz_str_list = xyz_obj.next_str()
                    with open(output_file, "w") as inp_obj:
                        if pref["use_chk_file"]:
                            inp_obj.write("%chk={}.chk\n".
                                          format(basename(pref["output"])
                                                 .split(".")[0]))
                        if pref["header_file"]:
                            with open(pref["header_file"], "r") as header_obj:
                                for read_line in header_obj:
                                    inp_obj.write(read_line)
                        else:
                            inp_obj.write(pref["header_str"])
                        for xyz in xyz_str_list:
                            inp_obj.write(xyz)
                        inp_obj.write("\n")
                        if pref["append_info"]:
                            inp_obj.write(pref["append_info"])
    else:
        with open(pref["output"], "w") as inp_obj:
            if pref["use_chk_file"]:
                inp_obj.write("%chk={}.chk\n".
                              format(basename(pref["output"]).split(".")[0]))
            if pref["header_file"]:
                with open(pref["header_file"], "r") as header_obj:
                    for read_line in header_obj:
                        inp_obj.write(read_line)
            else:
                inp_obj.write(pref["header_str"])
            if isinstance(pref["input"][0], str):    # list of str
                for xyz_info in pref["input"]:
                    inp_obj.write(xyz_info.rstrip() + "\n")
            else:    # list of list/tuple
                for xyz_info in pref["input"]:
                    inp_obj.write("{} {:.6f} {:.6f} {:.6f}\n".
                                  format(xyz_info[0], xyz_info[1],
                                         xyz_info[2], xyz_info[3]))
            inp_obj.write("\n")
            if pref["append_info"]:
                inp_obj.write(pref["append_info"])
    return result(0, pref)


def write_out_file(*args, **kwargs):
    """
    Returns:
        0 if succeed: (int)
    Args:
        out_file: output file with the file extension .out (str)
        input: xyz, gaussian input/output file, xyz_list, or atom_num
            (int, str, or list of list)
        header: header of job type, mem, nproc, keywords, charge and multiplicity (str)
        eng: electronic or potential energy (float)
        SCF_eng: electronic (SCF) or potential energy (float)
        MP2_eng: MP2 energy (float)
        CCSDT_eng: CCSD(T) energy (float)
        BSSE_eng: energy of counterpoise correction (float)
        freq_list: wavenumber list in cm^-1 from out2freq (list of float)
        inten_list: intensity list from out2inten (list of float)
        force_list: force list from out2force (list of list)
        pref: preference dict with above parameters as keys
    """
    pref = OrderedDict()
    pref["out_file"] = None
    pref["input"] = None
    pref["header"] = ""
    pref["eng"] = None
    pref["SCF_eng"] = None
    pref["MP2_eng"] = None
    pref["CCSDT_eng"] = None
    pref["BSSE_eng"] = None
    pref["freq_list"] = []
    pref["inten_list"] = []
    pref["force_list"] = []
    pref = update_pref(args, kwargs, pref)
    xyz_string = ""
    scf_eng_string = ""
    mp2_eng_string = ""
    ccsdt_eng_string = ""
    bsse_eng_string = ""
    force_string = ""
    freq_string = ""
    zpe_corrected_string = ""
    atom_num = 0
    if pref["input"]:
        if isinstance(pref["input"], (int, float)):  # Detect atom_num
            atom_num = int(pref["input"])
        else:
            xyz_list = xyz2list(pref["input"])
            atom_num = len(xyz_list)
            xyz_string = """\
                            Standard orientation:\n\
 ---------------------------------------------------------------------\n\
 Center     Atomic      Atomic             Coordinates (Angstroms)\n\
 Number     Number       Type             X           Y           Z\n\
 ---------------------------------------------------------------------\n"""
            for I0, xyz in enumerate(xyz_list):
                xyz_string += " {:>6d}        {:>3d}         {:>3d}     {:>11.6f} {:>11.6f} {:>11.6f}\n"\
                    .format(I0 + 1,
                            atomic_number[xyz[0]],
                            0,
                            float(xyz[1]),
                            float(xyz[2]),
                            float(xyz[3]))
    E0 = None
    if pref["eng"]:
        scf_eng_string = \
            " SCF Done:  E(X) = {:>15.9f}     A.U. after {:>4d} cycles\n"\
            .format(pref["eng"], 1)
        E0 = pref["eng"]
    if pref["SCF_eng"]:
        scf_eng_string = \
            " SCF Done:  E(X) = {:>15.9f}     A.U. after {:>4d} cycles\n"\
            .format(pref["SCF_eng"], 1)
        E0 = pref["SCF_eng"]
    if pref["MP2_eng"]:
        mp2_eng_string = \
            "E2 = No_data EUMP2 = {:1.14e}\n".format(pref["MP2_eng"])
        mp2_eng_string = mp2_eng_string.replace("e", "D")
        E0 = pref["MP2_eng"]
    if pref["CCSDT_eng"]:
        ccsdt_eng_string = "CCSD(T)= {}\n".format(pref["CCSDT_eng"])
        E0 = pref["CCSDT_eng"]
    if pref["BSSE_eng"]:
        bsse_eng_string = "BSSE energy = {}\n".format(pref["BSSE_eng"])
    if pref["force_list"]:
        force_string = """\
 Center     Atomic                   Forces (Hartrees/Bohr)\n\
 Number     Number              X              Y              Z\n\
 -------------------------------------------------------------------\n"""
        max_force = -999999
        rms_force = 0.0
        for I0, force in enumerate(pref["force_list"]):
            force_string += " {:>6d} {:>8d}        {:>14.9f} {:>14.9f} {:>14.9f}\n"\
                .format(I0,
                        atomic_number[xyz_list[I0][0]],
                        force[0],
                        force[1],
                        force[2])
            for tmp_force in force:
                if tmp_force > max_force:
                    max_force = tmp_force
                rms_force += tmp_force ** 2
        rms_force = sqrt(rms_force / float(atom_num * 3))
        force_string += """\
 -------------------------------------------------------------------\n\
 Cartesian Forces:  Max {:>15.9f} RMS {:>15.9f}\n""".format(max_force,
                                                            rms_force)
    if pref["freq_list"]:
        if atom_num == 0:
            atom_num = int((len(pref["freq_list"]) + 6) / 3)
        zpe = freq2zpe(pref["freq_list"], eng_unit="zpe_hartree")
        if E0:
            zpe_corrected_string = " Sum of electronic and zero-point Energies=         {:>13.6f}\n"\
                .format(E0 + zpe)
        freq_num = len(pref["freq_list"])
        freq_write_iter, remainder = divmod(freq_num, 3)
        for I0 in range(freq_write_iter):
            # freq_string += "                   {:>4d}                   {:>4d}                   {:>4d}\n"\
                # .format(I0*3+1, I0*3+2, I0*3+3)
            # freq_string += "                      A                      A                      A\n"
            freq_string += " Frequencies --  {:>10.4f}             {:>10.4f}             {:>10.4f}\n"\
                .format(pref["freq_list"][I0 * 3],
                        pref["freq_list"][I0 * 3 + 1],
                        pref["freq_list"][I0 * 3 + 2])
            if pref["inten_list"]:
                freq_string += " IR Inten    --  {:>10.4f}             {:>10.4f}             {:>10.4f}\n"\
                    .format(pref["inten_list"][I0 * 3],
                            pref["inten_list"][I0 * 3 + 1],
                            pref["inten_list"][I0 * 3 + 2])
        if remainder == 1:
            # freq_string += "                   {:>4d}\n".format(freq_num)
            # freq_string += "                      A\n"
            freq_string += " Frequencies --  {:>10.4f}\n"\
                .format(pref["freq_list"][freq_num - 1])
            if pref["inten_list"]:
                freq_string += " IR Inten    --  {:>10.4f}\n"\
                    .format(pref["inten_list"][freq_num - 1])
        elif remainder == 2:
            # freq_string += "                   {:>4d}                   {:>4d}\n"\
                # .format(freq_num-1, freq_num)
            # freq_string += "                      A                      A\n"
            freq_string += " Frequencies --  {:>10.4f}             {:>10.4f}\n"\
                .format(pref["freq_list"][freq_num - 2],
                        pref["freq_list"][freq_num - 1])
            if pref["inten_list"]:
                freq_string += " IR Inten    --  {:>10.4f}             {:>10.4f}\n"\
                    .format(pref["inten_list"][freq_num - 2],
                            pref["inten_list"][freq_num - 1])
    with open(pref["out_file"], "w") as out_obj:
        # For xyz_tools.xyz_type()
        out_obj.write(" Normal termination of Gaussian, Inc (resized)\n")
        out_obj.write(pref["header"])
        out_obj.write(" NAtoms= {:>4d}\n".format(atom_num))  # must have
        out_obj.write(xyz_string)
        if scf_eng_string:
            out_obj.write(scf_eng_string)
        if mp2_eng_string:
            out_obj.write(mp2_eng_string)
        if ccsdt_eng_string:
            out_obj.write(ccsdt_eng_string)
        if bsse_eng_string:
            out_obj.write(bsse_eng_string)
        if force_string:
            out_obj.write(force_string)
        if freq_string:
            out_obj.write(freq_string)
        if zpe_corrected_string:
            out_obj.write(zpe_corrected_string)
    return result(0, pref)


def gaussian_job(*args, **kwargs):
    """
    Returns:
        0 if succeed.
    Args:
        input: xyz_list, xyz_file, or inp_file (list of list or str)
        gaussian_exe: can be "g16" or "g09" (str)
        mem: overwrite memory usage in GB (int)
        option: overwrite job option (str)
        GAUSS_SCRDIR: path to GAUSS_SCRDIR (str)
        env: gaussian environment shell command (str)
        nproc: overwrite number of processors per job (int or None)
        preserve_scratch_file: True will preserve *.dat and *.F* files (bool)
        preserve_out: False will remove any Gaussian output files (bool)
        preserve_xyz: True will store the Xyz file (bool)
        convert_to_xyz: True will convert the results into Xyz format and clean
            all the results (bool)
        convert_frame_list: If convert_to_xyz is True, the frames will be
            chosen by this list (list or "all")
            to prevent gamess crashes with long path (bool)
        check_xyz: True will skip the job if the Xyz file exists (bool)
        check_out_exist: True will skip the job if the out file exists (bool)
        check_out: useful when convert to XYZ file, if check_out = True, convert to XYZ
                    will only applied of the output is "Normally terminated"
        skip_positive_eng: True will skip the positive energies (bool)
        timeout: set the timeout in sec for running the Gaussian (int)
        debug: if not None and is a folder, will copy the output files to that
            folder (str or bool)
        pref: preference which includes all parameters (dict)
    Note:
        input can be gaussian input file, xyz file, or xyz_list.
        If header is not None, it will always overwrite the input header.
        xyz_dict would be {"xyz_list": list of list ,
                           "eng": float, "zpe": float}

        If "output" == "input", it will name the output file based on the input
        file. If "input" is not an Gaussian input file, it will just name the
        output file as "input.out"

        If nproc != None, it will overwrite %NprocShared

        Set preserve_out = False, preserve_xyz = False, and
        convert_to_xyz = True for bh_func(). The xyz_list and eng will be
        stored in pref.

        If preserve_xyz = True, it will force to convert output file to the
        Xyz data.

        If convert_frame_list is an integer, it will apply an exponential
        sampling to the snapshots.
    """
    pref = OrderedDict()
    pref["input"] = None  # 0.inp
    pref["gaussian_exe"] = "g16"
    pref["header"] = None
    pref["mem"] = None
    pref["option"] = None
    pref["GAUSS_SCRDIR"] = "/tmp"
    pref["env"] = None
    pref["nproc"] = None
    pref["preserve_scratch_file"] = False
    pref["preserve_out"] = True
    pref["preserve_xyz"] = False
    pref["convert_to_xyz"] = False
    pref["convert_frame_list"] = [-1]
    pref["check_xyz"] = True
    pref["check_out_exist"] = True
    pref["check_out"] = False
    pref["skip_positive_eng"] = True
    pref["timeout"] = 18000
    pref["debug"] = False
    pref = update_pref(args, kwargs, pref)
    clean_inp = False
    exit_code = 0
    if pref["header"]:
        inp_header_str = header_str(pref["header"])
    else:
        inp_header_str = None
    if isinstance(pref["input"], (list, tuple)):
        abs_inp_file = mktemp("f",
                              use="mktemp",
                              mktemp_template="XXXXXX.inp",
                              mktemp_opt="-u")
        base = basename(abs_inp_file).split(".")[0]
        inp_dir = file_dir(abs_inp_file)
        if inp_header_str:
            with open(abs_inp_file, "w") as inp_obj:
                inp_obj.write(inp_header_str.rstrip()+"\n")
                for xyz in pref["input"]:
                    inp_obj.write("{} {} {} {}\n".format(xyz[0],
                                                         xyz[1],
                                                         xyz[2],
                                                         xyz[3]))
                inp_obj.write("\n")
            clean_inp = True
        else:
            print("Error! header is not exist! Quit!")
            return result(-1, pref)
    elif isinstance(pref["input"], str):
        ext = basename(pref["input"]).split(".")[-1].lower()
        if ext == "xyz":
            abs_inp_file = mktemp("f",
                                  use="mktemp",
                                  mktemp_template="XXXXXX.inp",
                                  mktemp_opt="-u")
            inp_dir = file_dir(abs_inp_file)
            base = basename(abs_inp_file).split(".")[0]
            xyz_list = xyz2list(pref["input"])
            if inp_header_str:
                with open(abs_inp_file, "w") as inp_obj:
                    inp_obj.write(inp_header_str.rstrip()+"\n")
                    for xyz in xyz_list:
                        inp_obj.write("{} {} {} {}\n".format(xyz[0],
                                                             xyz[1],
                                                             xyz[2],
                                                             xyz[3]))
                    inp_obj.write("\n")
                clean_inp = True
            else:
                print("Error! header is not exist! Quit!")
                return result(-1, pref)
        elif ext in ("inp", "gjf"):
            abs_inp_file = full_path(pref["input"])
            inp_dir = file_dir(abs_inp_file)
            base = basename(abs_inp_file).split(".")[0]
    abs_out_file = "{}/{}.out".format(inp_dir, base)
    abs_running_file = "{}/{}.out.running".format(inp_dir, base)
    abs_xyz_file = "{}/{}.xyz".format(inp_dir, base)
    if pref["check_xyz"] and isfile(abs_xyz_file):
        print("Detected \033[1;34m{}\033[0m. Skip!"
              .format(abs_xyz_file))
        return result(0, pref)
    if pref["check_out_exist"] and isfile(abs_out_file):
        print("Detected \033[1;34m{}\033[0m. Skip!"
              .format(abs_out_file))
        return result(0, pref)

    gauss_scrdir = mktemp("d", use="mktemp", tmp_dir=pref["GAUSS_SCRDIR"])
    # cmd("export GAUSS_SCRDIR={}".format(gauss_scrdir))
    if pref["nproc"]:
        nproc_line = shell_cmd("grep -a -i '%Nproc' {}".format(abs_inp_file))
        if nproc_line:
            shell_cmd("sed -i 's/{}/%NprocShared={}/g' {}"
                      .format(nproc_line[0].rstrip(),
                              pref["nproc"],
                              abs_inp_file))
    if pref["mem"]:
        mem_line = shell_cmd("grep -a -i '%mem=' {}".format(abs_inp_file))
        if mem_line:
            if "%" in pref["mem"] and "=" in pref["mem"]:
                shell_cmd("sed -i 's/{}/{}/g' {}".format(mem_line[0].rstrip(),
                                                         pref["mem"],
                                                         abs_inp_file))
            else:
                shell_cmd("sed -i 's/{}/%mem={}/g' {}"
                          .format(mem_line[0].rstrip(),
                                  pref["mem"],
                                  abs_inp_file))
    if pref["option"]:
        opt_line = shell_cmd("grep -a '#' {}".format(abs_inp_file))
        if opt_line:
            if "#" in pref["option"]:
                shell_cmd("sed -i 's/{}/{}/g' {}"
                          .format(opt_line[0].rstrip().replace("/", r"\/").replace("*", r"\*"),
                                  pref["option"].rstrip().replace("/", r"\/").replace("*", r"\*"),
                                  abs_inp_file))
            else:
                shell_cmd("sed -i 's/{}/# {}/g' {}"
                          .format(opt_line[0].rstrip().replace("/", r"\/").replace("*", r"\*"),
                                  pref["option"].rstrip().replace("/", r"\/").replace("*", r"\*"),
                                  abs_inp_file))
    if pref["env"]:
        exe_cmd = "{}; export GAUSS_SCRDIR={}; cd {}; {} < {} > {}".format(pref["env"],
                                                                           gauss_scrdir,
                                                                           inp_dir,
                                                                           pref["gaussian_exe"],
                                                                           abs_inp_file,
                                                                           abs_running_file)
        print_cmd = "{}; export GAUSS_SCRDIR={}; cd {}; {} < \033[1;34m{}\033[0m > \033[1;34m{}\033[0m".format(pref["env"],
                                                                                                               gauss_scrdir,
                                                                                                               inp_dir,
                                                                                                               pref["gaussian_exe"],
                                                                                                               abs_inp_file,
                                                                                                               abs_running_file)
    else:
        exe_cmd = "export GAUSS_SCRDIR={}; cd {}; {} < {} > {}".format(gauss_scrdir,
                                                                       inp_dir,
                                                                       pref["gaussian_exe"],
                                                                       abs_inp_file,
                                                                       abs_running_file)
        print_cmd = "export GAUSS_SCRDIR={}; cd {}; {} < \033[1;34m{}\033[0m > \033[1;34m{}\033[0m".format(gauss_scrdir,
                                                                                                           inp_dir,
                                                                                                           pref["gaussian_exe"],
                                                                                                           abs_inp_file,
                                                                                                           abs_running_file)
    print("\033[1;33mRunning\033[0m: ", print_cmd)
    start_time = now()
    # host = shell_cmd("hostname")[0]
    # if "cn" in host:  # For Taiwania
        # shell_cmd(exe_cmd, pref["timeout"])
    # else:
    job_cmd(exe_cmd, pref["timeout"])
    print("\033[1;33mFinished\033[0m: \033[1;34m{}\033[0m takes \033[1;36m{}\033[0m sec"
          .format(abs_inp_file,
                  delta_time(start_time, "sec")))
    # cmd(exe_cmd)
    shell_cmd("mv -f {} {}".format(abs_running_file, abs_out_file))
    if clean_inp:
        shell_cmd("rm -rf {}".format(abs_inp_file))
        exit_code = 0
    if pref["debug"] and isinstance(pref["debug"], str):
        shell_cmd("mkdir -p {}".format(pref["debug"]))
        print("Debug mode: copy \033[1;34m{}\033[0m to \033[1;34m{}\033[0m"
              .format(abs_out_file, pref["debug"]))
        shell_cmd("cp -rf {} {}".format(abs_out_file, pref["debug"]))
    if pref["convert_to_xyz"] or pref["preserve_xyz"]:
        normal_count = shell_cmd("grep -a 'Normal termination' {} | wc -l"
                                 .format(abs_out_file))[0].strip()
        error_count = shell_cmd("grep 'Erro' {} | grep -v 'RMS Error'| wc -l"
                                .format(abs_out_file))[0].strip()
        if int(normal_count) == 0 or int(error_count) > 0 and pref["check_out"]:
            print("\033[1;31mNo normal termination in \033[1;34m{}\033[1;31m. error=\033[1;36m{}\033[1;31m, normal=\033[1;36m{}\033[1;31m. Skip!\033[0m"
                  .format(abs_out_file, error_count, normal_count))
            if isinstance(pref["input"], (list, tuple)):
                pref["xyz_list"] = [pref["input"]]
            else:
                pref["xyz_list"] = []
            pref["eng"] = ["nan"]
        else:
            try:
                if isinstance(pref["convert_frame_list"], int):
                    eng_list = out2eng(abs_out_file,
                                       frame_list="all")
                    if isinstance(eng_list, float):
                        eng_list = [eng_list]
                        xyz_list = [g09_out2xyz(abs_out_file, frame_list=[-1])]
                        force_list = [out2force(abs_out_file, frame_list=[-1])]
                    else:
                        convert_frame_list = \
                            exp_sample_id_list(len(eng_list),
                                               pref["convert_frame_list"],
                                               -4,
                                               1)
                        fix_convert_frame_list = []
                        fix_eng_list = []
                        for check_eng_id in convert_frame_list:
                            if pref["skip_positive_eng"] and eng_list[check_eng_id] > 0.0:
                                print("\033[1;31mWarning! Detect positive energy in frame \033[1;36m{}\033[1;31m.Skip!\033[0m"
                                      .format(check_eng_id))
                            else:
                                fix_convert_frame_list.append(check_eng_id)
                                fix_eng_list.append(eng_list[check_eng_id])
                        eng_list = fix_eng_list
                        xyz_list = g09_out2xyz(abs_out_file,
                                               frame_list=fix_convert_frame_list)
                        # try:
                        force_list = out2force(abs_out_file,
                                               frame_list=fix_convert_frame_list)
                        # except:
                            # shell_cmd("cp -rf {} ~/fail_out".format(abs_out_file))
                        if len(fix_convert_frame_list) == 1:
                            xyz_list = [xyz_list]
                            if force_list:
                                force_list = [force_list]
                else:
                    eng_list = out2eng(abs_out_file,
                                       frame_list="all")
                    if isinstance(eng_list, float):
                        eng_list = [eng_list]
                    eng_num = len(eng_list)
                    if pref["convert_frame_list"] == "all":
                        xyz_list = g09_out2xyz(abs_out_file,
                                               frame_list="all")
                        force_list = out2force(abs_out_file,
                                               frame_list="all")
                    else:
                        fix_convert_frame_list = []
                        fix_eng_list = []
                        for check_eng_id in pref["convert_frame_list"]:
                            if check_eng_id >= eng_num:
                                print("\033[1;31mFrame \033[1;36m{}\033[1;31m does not exist. Skip! \033[0m"
                                      .format(check_eng_id))
                            elif pref["skip_positive_eng"] and eng_list[check_eng_id] > 0.0:
                                print("\033[1;31mWarning! Detect positive Eng in frame \033[1;36m{}\033[1;31m.Skip!\033[0m"
                                      .format(check_eng_id))
                            else:
                                fix_convert_frame_list.append(check_eng_id)
                                fix_eng_list.append(eng_list[check_eng_id])
                        eng_list = fix_eng_list
                        xyz_list = g09_out2xyz(abs_out_file,
                                               frame_list=fix_convert_frame_list)
                        force_list = out2force(abs_out_file,
                                               frame_list=fix_convert_frame_list)
                        if len(fix_convert_frame_list) == 1:
                            xyz_list = [xyz_list]
                            if force_list:
                                force_list = [force_list]
                if len(force_list) == len(xyz_list):
                    tmp_xyz_list = []
                    for I0, xyz in enumerate(xyz_list):
                        tmp_xyz_list.append(join_2d_list(xyz, force_list[I0]))
                    xyz_list = tmp_xyz_list
                pref["xyz_list"] = xyz_list
                pref["eng"] = eng_list
                if eng_list and pref["preserve_xyz"]:
                    zpe = out2zpe(abs_out_file)
                    with Xyz(abs_xyz_file, "w") as xyz_obj:
                        for I0, eng in enumerate(eng_list[:-1]):
                            xyz_info = "{} eng= {} Properties= species:S:1:pos:R:3:force:R:3".format(I0, eng)
                            xyz_obj.write(xyz_list[I0], xyz_info)
                        zpe_id = len(eng_list) - 1
                        if zpe:
                            xyz_info = "{} eng= {} zpe= {} Properties= species:S:1:pos:R:3:force:R:3".format(zpe_id,
                                                                                                             eng_list[-1],
                                                                                                             zpe)
                        else:
                            xyz_info = "{} eng= {} Properties= species:S:1:pos:R:3:force:R:3".format(zpe_id, eng_list[-1])
                        xyz_obj.write(xyz_list[zpe_id], xyz_info)
            except Exception as e:
                print(f"\033[1;31mWarning! There may be an error in exporting to XYZ: {str(e)} \033[1;36m\033[1;31m.Skip!\033[0m")
                pref["xyz_list"] = []
                pref["eng"] = ["nan"]
                return result(exit_code, pref)

    if not pref["preserve_out"]:
        shell_cmd("rm -rf {}".format(abs_out_file))
    if pref["preserve_scratch_file"]:
        shell_cmd("cp -rf {}/* {}".format(gauss_scrdir, inp_dir))
    shell_cmd("rm -rf {}".format(gauss_scrdir))
    return result(exit_code, pref)


def gaussian_job_old(*args, **kwargs):
    """
    Returns:
        xyz_dict: (dict)
    Args:
        input: xyz_list or xyz_file/inp_file/out_file (list of list or str)
        output: gaussian output file (str)
        gaussian_exe: can be "g16" or "g09" (str)
        nproc: overwrite number of processors per job (int or None)
        mem: overwrite memory usage in GB (int)
        option: overwrite job option (str)
        header: header_file or header_str (str)
        use_chk_file: True will generate check point file (bool)
        GAUSS_SCRDIR: path to GAUSS_SCRDIR (str)
        gaussian_env: gaussian environment shell command (str)
        check_normal: True will check if the output file is normal terminateion
            (bool)
        check_img_freq: True will check if the output file contains imaginary
            frequencies (bool)
        check_xyz: True will check if the output file contains coordinates
            (bool)
        resume_last_step: True will check the output file and try to rebuild
            the input file from the failed output file (bool)
        pref: preference which includes all parameters (dict)
    Note:
        input can be gaussian input file, xyz file, or xyz_list.
        If header is not None, it will always overwrite the input header.
        gaussian_env must be assigned for gaussian_bin and gaussian.profile.
        xyz_dict would be {"xyz_list": list of list ,
                           "eng": float, "zpe": float}

        If "output" == "input", it will name the output file based on the input
        file. If "input" is not an Gaussian input file, it will just name the
        output file as "input.out"

        If nproc != None, it will overwrite %NprocShared
    """
    pref = OrderedDict()
    pref["input"] = None  # 0.inp
    pref["output"] = "input"
    pref["gaussian_exe"] = "g16"
    pref["nproc"] = None
    pref["header"] = None
    pref["use_chk_file"] = False
    pref["GAUSS_SCRDIR"] = "/tmp"
    pref["gaussian_env"] = "export g09root=/home/software/g09-d01;source /home/software/g09-d01/g09/bsd/g09.profile;alias gaussian_bin=/home/software/g09-d01/g09/g09"
    pref["check_normal"] = True
    pref["check_img_freq"] = True
    pref["check_xyz"] = True
    pref["resume_last_step"] = False
    pref = update_pref(args, kwargs, pref)
    clean_inp = False
    if pref["input"]:
        if not pref["header"] and \
                isfile(pref["input"]) and \
                xyz_type(pref["input"]) == "g09_out":
            pref["header"] = out2header(pref["input"], header_type="g09_inp")
    else:
        return result(1, pref)
    if pref["header"]:  # Always use xyz2inp to overwrite header
        gaussian_input = escaped_file_name("{}.inp".format(pref["input"].split(".")[0]))
        xyz2inp(pref["input"],
                output=gaussian_input,
                header=pref["header"],
                use_chk_file=pref["use_chk_file"])
        clean_inp = True
    else:
        if isfile(pref["input"]) and xyz_type(pref["input"]) == "g09_inp":
            gaussian_input = escaped_file_name(pref["input"])
        else:
            print("Warning! header information is missing in \033[1;34m{}\033[0m"
                  .format(pref["input"]))
            return result(1, pref)
    shell_cmd(pref["gaussian_env"])
    gauss_scrdir = mktemp("d", use="mktemp", tmp_dir=pref["GAUSS_SCRDIR"])
    shell_cmd("export GAUSS_SCRDIR={}".format(gauss_scrdir))
    if pref["output"] == "input" or not pref["output"]:
        pref["output"] = "{}.out".format(gaussian_input.split(".")[0])
    if pref["nproc"]:
        nproc_line = shell_cmd("grep -a -i '%Nproc' {}".format(gaussian_input))
        if nproc_line:
            shell_cmd("sed -i 's/{}/%NprocShared={}/g' {}"
                     .format(nproc_line[0].rstrip(),
                        pref["nproc"],
                        gaussian_input))
    if pref["mem"]:
        mem_line = shell_cmd("grep -a -i '%mem=' {}".format(gaussian_input))
        if mem_line:
            if "%" in pref["mem"] and "=" in pref["mem"]:
                shell_cmd("sed -i 's/{}/{}/g' {}".format(mem_line[0].rstrip(),
                                                         pref["mem"],
                                                         gaussian_input))
            else:
                shell_cmd("sed -i 's/{}/%mem={}/g' {}".format(mem_line[0].rstrip(),
                                                              pref["mem"],
                                                              gaussian_input))
    if pref["option"]:
        opt_line = shell_cmd("grep -a '#' {}".format(gaussian_input))
        if opt_line:
            if "#" in pref["option"]:
                shell_cmd("sed -i 's/{}/{}/g' {}"
                          .format(opt_line[0].rstrip().replace("/", r"\/").replace("*", r"\*"),
                                  pref["option"].rstrip().replace("/", r"\/").replace("*", r"\*"),
                                  gaussian_input))
            else:
                shell_cmd("sed -i 's/{}/# {}/g' {}"
                          .format(opt_line[0].rstrip().replace("/", r"\/").replace("*", r"\*"),
                                  pref["option"].rstrip().replace("/", r"\/").replace("*", r"\*"),
                                  gaussian_input))
    shell_cmd("{} < {} > {}.running".format(pref["gaussian_exe"],
                                            gaussian_input,
                                            pref["output"]))
    shell_cmd("rm -rf {}".format(gauss_scrdir))
    shell_cmd("mv -f {}.running {}".format(pref["output"], pref["output"]))
    if clean_inp:
        shell_cmd("rm -rf {}".format(gaussian_input))
    if pref["resume_last_step"]:
        exit_code = check_out(pref["output"],
                              check_normal=pref["check_normal"],
                              check_img_freq=pref["check_img_freq"],
                              check_xyz=pref["check_xyz"],
                              resume_last_step=pref["resume_last_step"])
    else:
        exit_code = 0

    # eng = out2eng(pref["output"])
    # zpe = out2zpe(pref["output"])
    # xyz_list = xyz2list(pref["output"])
    # return result({"xyz_list": xyz_list, "eng": eng, "zpe": zpe}, pref)
    return result(exit_code, pref)


def g09_stat(*args, **kwargs):
    """
    Returns:
        g09_dict = {"eng": eng_list, "xyz": xyz_list, "zpe", "force", ...}: (dict)
    Args:
        input: gaussian output file (str)
        frame_list: list of opt steps to be extracted (list or "all")
        find_zpe: True will compute zero-point corrected energy (bool)
        find_freq: True will extract 3N-6 frequencies (bool)
        find_inten: True will extract 3N-6 intensities (bool)
        find_force: True will extract forces (bool)
        find_src: True will record the path of input and the frame id (bool)
        force_multiply_factor: default is 1.8897261 to convert the force unit
            of a Gaussian output file from Hartree/Bohr to Hartree/Angstrom
            (float)
        pref: preference dict with above parameters as keys
    Note:
        eng, xyz, and force will always be extracted as multi-frames even only
        one frame is presented:
            g09_dict["eng"] = [eng1, eng2, eng3,...]
            g09_dict["xyz"] = [xyz_list1, xyz_list2, xyz_list3,...]
            g09_dict["force"] = [force_list1, force_list2, force_list3,...]
            g09_dict["src"] = ["x.out:0", "x.out:1", ...]
            g09_dict["zpe"] = zpe_value
            g09_dict["freq"] = [freq1, freq1, ...freq3N-6]
            g09_dict["inten"] = "inten1, inten2, ...intel3N-6"
            g09_dict["opt"] = True if "Optimization completed" keyword in the
            output file.\n\n

        Original force unit in the Gaussian output file is Hartree/Bohr_radius.
        To convert it to Hartree/A, forces will be automatically multiplied by
        1/Bohr_radius_in_Angstrom or 1.0 / 0.5291772105638411 by default.
        To avoid the conversion, set force_multiply_factor to 1.0.
    """
    pref = OrderedDict()
    pref["input"] = "0.out"
    pref["frame_list"] = [-1]
    pref["find_zpe"] = False
    pref["find_freq"] = False
    pref["find_inten"] = False
    pref["find_force"] = False
    pref["find_src"] = False
    pref["find_dipole_der"] = False
    pref["force_multiply_factor"] = inv_bohr_radius_ang
    pref = update_pref(args, kwargs, pref)
    g09_dict = {"type": xyz_type(pref["input"])}
    if g09_dict["type"] == "g09_out":
        check_opt = shell_cmd("grep -a 'Optimization completed' {}"
                              .format(pref["input"]))
        opt_stat = False
        if check_opt:
            opt_stat = True
        eng = out2eng(pref["input"], frame_list=pref["frame_list"])
        if isinstance(eng, (int, float)):
            if pref["frame_list"] == "all":
                pref["frame_list"] = [0]
            eng = [eng]
        g09_dict["eng"] = eng
        xyz_list = xyz2list(pref["input"], frame_list=pref["frame_list"])
        if isinstance(xyz_list[0][0], str):
            xyz_list = [xyz_list]
        g09_dict["xyz"] = xyz_list
        if pref["find_force"]:
            force_list = out2force(pref["input"],
                                   frame_list=pref["frame_list"],
                                   force_multiply_factor=pref["force_multiply_factor"])
            if force_list:
                if isinstance(force_list[0][0], (int, float)):
                    force_list = [force_list]
                g09_dict["force"] = force_list
        if pref["find_zpe"]:
            zpe = out2zpe(pref["input"])
            if zpe:
                g09_dict["zpe"] = zpe
        if pref["find_freq"]:
            freq_list = out2freq(pref["input"])
            if freq_list:
                g09_dict["freq"] = freq_list
        if pref["find_inten"]:
            inten_list = out2inten(pref["input"])
            if inten_list:
                g09_dict["inten"] = inten_list
        if pref["find_dipole_der"]:
            dipole_der_list = out2dipole_der(pref["input"])
            if dipole_der_list:
                g09_dict["dipole_der"] = dipole_der_list
        if pref["find_src"]:
            src_list = []
            for I0 in range(len(eng)):
                src_list.append("{}:{}".format(pref["input"], I0))
            g09_dict["src"] = src_list
        g09_dict["opt"] = opt_stat
    elif g09_dict["type"] == "g09_inp":
        inp_dict = g09_inp2xyz(pref["input"], out_type="dict")
        g09_dict["xyz"] = inp_dict["xyz"]
        g09_dict["header"] = inp_dict["header"]
        if pref["find_src"]:
            g09_dict["src"] = "{}:0".format(pref["input"])
    return result(g09_dict, pref)


# pref_tools {
def update_pref(args=None, kwargs=None, pref=None):
    """
    Returns:
        pref: preference dict
    Args:
        args: tuple
        kwargs: dict
        pref: preference dict
    Note:
        kwargs will overwrite args.
        Once pref is used, both args and kwargs will be ignore.
    """
    if "pref" in kwargs:
        if "dict" not in type(kwargs["pref"]).__name__.lower():
            print("Error! pref={} is not a dict!".format(kwargs["pref"]))
        for key in kwargs["pref"]:
            pref[key] = kwargs["pref"][key]
        pref["use_pref"] = True
    else:
        pref_keys = list(pref.keys())
        for id, arg in enumerate(args):
            pref[pref_keys[id]] = arg
        for key in kwargs:
            pref[key] = kwargs[key]
        pref["use_pref"] = False
    return pref


def result(results=None, pref={"return": None}):
    """
    Returns:
        results: if pref["return"] == "pref", it will return preference dict
    Args:
        results: any (normal esults)
        pref: preference dict
    Note:
        0 for a successful running.
        1 or other non-zero values for a failed running.
    """
    if pref["use_pref"]:
        pref["return"] = results
        return pref
    else:
        return results


# } pref_tools
def main():
    """ main """


if __name__ == "__main__":
    main()
