#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# ==============================================================
# Copyright(c) 2017-, Po-Jen Hsu (clusterga@gmail.com)
#              2018-, Chun-Jung Huang (b10304106@mail.ntust.edu.tw)
# See the README file in the top-level directory for license.
# ==============================================================
# Creation Time : 20170627 11:40:01
# Modification Time : 20180814 10:26:01
# ==============================================================
from collections import OrderedDict
from os import environ
from os.path import basename
from hash_tools import hash_str
from os_tools import full_path, single_cmd
from std_tools import std_err
from xyz_tools import Xyz, atomic_number_list, xyz2list, xyz_str_list

minimization_DFTB = " $CONTRL\n\
   RUNTYP=OPTIMIZE NPRINT=-5 ICHARG={molecular_charge} MULT={multiplicity} MAXIT=200\n\
 $END\n\
   nzvar=1\n\
 $ZMAT\n\
 $END\n\
    DLC=.t.\n\
   AUTO=.t.\n\
 $STATPT\n\
   projct=.f.\n\
   method=rfo\n\
   NSTEP={nstep}\n\
   RMIN=0.0001\n\
   OPTTOL=5.0D-7\n\
 $END\n\
 $SYSTEM MWORDS=250 $END\n\
 $SCF\n\
   dirscf=.t.\n\
   diis=.t.\n\
   damp=.t.\n\
   CONV=5.0D-7\n\
 $END\n\
 $BASIS {basis} $END\n\
 $DFTB\n\
   NDFTB=3\n\
   DAMPXH=.t.\n\
   DAMPEX=4.1\n\
   itypmx=2\n\
 $END\n\
 $DFTBSK\n\
{DFTBSK_part}\
\n\
\n\
 $END\n\
 $DATA\n\
 \n\
 C1\n\
{xyz_part}\
 $END\n"

single_point_DFTB = " $CONTRL RUNTYP=ENERGY NPRINT=-5 ICHARG={molecular_charge} \
MULT={multiplicity} MAXIT=200 $END\n\
 $SYSTEM MWORDS=125 $END\n\
 $STATPT NSTEP=1000 RMIN=0.0001 OPTTOL=5.0D-7 $END\n\
 $SCF CONV=5.0D-7 $END\n\
 $BASIS {basis} $END\n\
 $DFTB\n\
  NDFTB=3\n\
  DAMPXH=.t.\n\
  DAMPEX=4.1\n\
  itypmx=2\n\
 $END\n\
 $DFTBSK\n\
{DFTBSK_part}\
\n\
\n\
 $END\n\
 $DATA\n\
\n\
C1\n\
{xyz_part}\
 $END"

TS = " $CONTRL\n\
   RUNTYP=SADPOINT NPRINT=-5 ICHARG={molecular_charge} MULT={multiplicity} MAXIT=200\n\
 $END\n\
   nzvar=1\n\
 $ZMAT\n\
 $END\n\
    DLC=.t.\n\
   AUTO=.t.\n\
 $STATPT\n\
   projct=.f.\n\
   IHREP=1\n\
   HESS=CALC\n\
   method=rfo\n\
   NSTEP=1000\n\
 $END\n\
 $SYSTEM MWORDS=250 $END\n\
 $SCF\n\
   dirscf=.t.\n\
   diis=.t.\n\
   damp=.t.\n\
 $END\n\
 $BASIS {basis} $END\n\
 $DFTB\n\
   NDFTB=3\n\
   DAMPXH=.t.\n\
   DAMPEX=4.1\n\
   itypmx=2\n\
 $END\n\
 $DFTBSK\n\
{DFTBSK_part}\
\n\
\n\
 $END\n\
 $DATA\n\
 \n\
 C1\n\
{xyz_part}\
 $END\n"
vibration = " $CONTRL RUNTYP=HESSIAN NPRINT=-5 ICHARG={molecular_charge} \
MULT={multiplicity} MAXIT=200 $END\n\
 $SYSTEM MWORDS=125 $END\n\
 $STATPT NSTEP=1000 RMIN=0.0001 OPTTOL=5.0D-7 $END\n\
 $SCF CONV=5.0D-7 $END\n\
 $BASIS {basis} $END\n\
 $DFTB\n\
  NDFTB=3\n\
  DAMPXH=.t.\n\
  DAMPEX=4.1\n\
  itypmx=2\n\
 $END\n\
 $DFTBSK\n\
{DFTBSK_part}\
\n\
\n\
 $END\n\
 $DATA\n\
\n\
C1\n\
{xyz_part}\
 $END"
minimization_B3LYP = " $CONTRL SCFTYP=RHF RUNTYP=OPTIMIZE ICHARG={molecular_charge} \
MULT={multiplicity} MAXIT=200 $END\n\
 $SYSTEM TIMLIM=600000 MWORDS=125 $END\n\
 $STATPT NSTEP=1000 $END\n\
 $DFT DFTTYP=B3LYP $END\n\
 $BASIS {basis} $END\n\
 $DATA\n\
 \n\
 C1\n\
{xyz_part}\
 $END"
#noted that the parameters of B3LYP+D about this D3 correction is based on g09 GD3.
#reference: http://wild.life.nctu.edu.tw/~jsyu/compchem/g09/g09ur/k_dft.htm
minimization_B3LYP_D = " $CONTRL SCFTYP=RHF RUNTYP=OPTIMIZE ICHARG={molecular_charge} \
MULT={multiplicity} MAXIT=200 $END\n\
 $SYSTEM TIMLIM=600000 MWORDS=125 $END\n\
 $STATPT NSTEP=1000 $END\n\
 $DFT DFTTYP=B3LYP DC=.t. DCS8=1.7030 DCSR=1.2610 IDCVER=3 $END\n\
 $BASIS {basis} $END\n\
 $DATA\n\
 \n\
 C1\n\
{xyz_part}\
 $END"
minimization_wB97XD = " $CONTRL SCFTYP=RHF RUNTYP=OPTIMIZE ICHARG={molecular_charge} \
MULT={multiplicity} $END\n\
 $SYSTEM TIMLIM=600000 MWORDS=125 $END\n\
 $STATPT NSTEP=1000 $END\n\
 $DFT DFTTYP=wB97X-D $END\n\
 $BASIS {basis} $END\n\
 $DATA\n\
 \n\
 C1\n\
{xyz_part}\
 $END"
minimization_MP2 = " $CONTRL SCFTYP=RHF RUNTYP=OPTIMIZE MPLEVEL=2 ICHARG={molecular_charge} MULT={multiplicity} $END\n\
 $SYSTEM TIMLIM=600000 MWORDS=125 $END\n\
 $SCF DIRSCF=.t. $END\n\
 $STATPT NSTEP=1000 $END\n\
 $DFT DFTTYP=B2PLYP CMP2=1.0 CHF=0.0$END\n\
 $BASIS {basis} $END\n\
 $DATA\n\
 \n\
 C1\n\
{xyz_part}\
 $END"

input_template_dict = {
    "min": minimization_DFTB,
    "sp_dftb": single_point_DFTB,
    "ts": TS,
    "vib": vibration,
    "min_B3LYP": minimization_B3LYP,
    "min_B3LYP+D": minimization_B3LYP_D,
    "min_wB97XD": minimization_wB97XD,
    "min_MP2": minimization_MP2,
}

basis_dict = {
    "DFTB": "GBASIS=DFTB",
    "6-311+G(2d,p)": "GBASIS=N311 NGAUSS=6 NDFUNC=2 NPFUNC=1 DIFFSP=.t.",
    "6-311+G(d,p)": "GBASIS=N331 NGAUSS=6 NDFUNC=1 NPFUNC=1 DIFFSP=.t.",
    "6-31+G(d)": "GBASIS=N31 NGAUSS=6 NDFUNC=1 DIFFSP=.t.",
}


def gamess_vibmin(*args, **kwargs):
    """
    Returns:
        xyz_list: (list of list)
        eng: (float)
        freq_list: (list of float)
        inten_list: (list of float)
    Args:
        input: xyz_list or xyz_file (list of list or str)
        parm_loc: for locating parameter files(.skf file) (str)
        gamess_loc: for gamess executable file (str)
        verno: version number of your GAMESS (str)
        mode: "min" for DFTB minimization,
              "vib" for vibrational frequencies and intensities,
              "ts"  for TS search,
              "min_B3LYP"   for DFT B3LYP minimization,
              "min_B3LYP+D" for DFT B3LYP minimization with D3 correction,
              "min_wB97XD"  for DFT wB97XD minimization,
              "min_MP2" is still under construction but you could try it
        basis: Currently available basis are listed below
               If you think that there is other basis you would use
               frequently, please report to the developer
               "DFTB",
               "6-31+G(d)",
               "6-311+G(d,p)",
               "6-311+G(2d,p)",
        nproc: you can only assign 1 now. It is still under construction
        molecular_charge: for ICHARG variable in .inp file
        multiplicity: mutiplicity calculated by 2S+1
        preserve_inp: store the .inp file and other output files of GAMESS
        debugging_mode: show the return value
        tmp: the path of the .inp file and other output files of GAMESS
        pref: preference which includes all parameters (dict)
    Note:
        It is still an unoptimized version but it should work stably now.
        If you find any bug when you run it, please report it to the developer.
    """
    pref = OrderedDict()
    pref["input"] = None
    pref["parm_loc"] = None
    pref["gamess_loc"] = None
    pref["nstep"] = 1000
    pref["verno"] = "00"
    pref["mode"] = "min"
    pref["basis"] = "DFTB"
    pref["nproc"] = 1
    pref["molecular_charge"] = "0"
    pref["multiplicity"] = "1"
    pref["custom_mpi_cmd"] = None
    pref["preserve_inp"] = False
    pref["debugging_mode"] = True
    pref["tmp"] = "/tmp"
    pref = update_pref(args, kwargs, pref)
    dft_mode = ["min_B3LYP", "min_B3LYP+D", "min_wB97XD", "min_MP2"]
    dftb_mode = ["min", "vib", "ts", "sp_dftb"]
    #Default basis for DFT
    if pref["mode"] == "min_B3LYP":
        pref["basis"] = "6-31+G(d)"
    elif pref["mode"] == "min_B3LYP+D":
        pref["basis"] = "6-31+G(d)"
    elif pref["mode"] == "min_MP2":
        pref["basis"] = "6-311+G(2d,p)"
    elif pref["mode"] == "min_wB97XD":
        pref["basis"] = "6-311+G(2d,p)"
    pref = update_pref(args, kwargs, pref)
    if pref["nproc"] == "all":
        nproc = int(single_cmd("lscpu |grep 'CPU(s)' | head -n 1 | awk '{print $2}'")[0])
    else:
        nproc = pref["nproc"]
    if isinstance(pref["input"], (list, tuple)):
        base_name = pref["mode"]
    else:
        base_name = basename(pref["input"]).split(".")[0]
        pref["input"] = xyz2list(pref["input"])
    hash_dir = "{}/{}".format(pref["tmp"], hash_str(algorithm="md5"))
    single_cmd("rm -rf {}; mkdir -p {}".format(hash_dir, hash_dir))
    #hssend = "HSSEND=.f."
    #if pref["mode"] == " ":
    #    hssend = "HSSEND=.t."
    #Write the $DFTBSK part for .inp file
    DFTBSK = ""
    atom_exist = []
    for item in pref["input"]:
        if item[0] in atom_exist:
            pass
        else:
            atom_exist.append(item[0])
    for x_mol in atom_exist:
        for y_mol in atom_exist:
            DFTBSK = DFTBSK + "   " + x_mol + " " + y_mol + ' "'\
            + pref["parm_loc"] + "/" + x_mol + "-" + y_mol + '.skf"\n'
    #Write the xyz_part for .inp file
    atom_number = 0
    atom_weight_dic = {"H": "1.0", "C": "6.0", "O": "8.0", "N": "7.0",
                       "NA": "11.0","S": "16.0", "Na": "11.0"}
    xyz_part = ""
    for item in pref["input"]:
        xyz_part = (xyz_part + " " + item[0] + "  " + atom_weight_dic[item[0]]
        + "    " + str(item[1]) + "    " + str(item[2]) + "    " + str(item[3])
        + "\n")
        atom_number += 1
    inp_file = "{}/{}.inp".format(hash_dir, base_name)
    std_file = "{}/{}".format(hash_dir, base_name)
    dat_file = "{}/{}.dat".format(hash_dir, base_name)
    out_file = "{}/{}.out".format(hash_dir, base_name)
    if nproc > 1:
        if pref["custom_mpi_cmd"]:
            mpi_cmd = pref["custom_mpi_cmd"]
        elif "PBS_NODEFILE" in environ:
            host_file = environ["PBS_NODEFILE"]
            mpi_cmd = "mpirun -hostfile {} -n {}".format(host_file, nproc)
        else:
            host_name = single_cmd("hostname")[0].strip()
            mpi_cmd = "mpirun -H {}:{} -n {}".format(host_name, nproc, nproc)
    else:
        mpi_cmd = ""
    # Write input file
    if pref["mode"] in dft_mode:
        with open(inp_file, "w") as inp_file_obj:
            inp_file_obj.write(input_template_dict[pref["mode"]].format(molecular_charge=pref["molecular_charge"],
                                                                        multiplicity=pref["multiplicity"],
                                                                        basis=basis_dict[pref["basis"]],
                                                                        xyz_part=xyz_part))
    elif pref["mode"] in dftb_mode:
        with open(inp_file, "w") as inp_file_obj:
            inp_file_obj.write(input_template_dict[pref["mode"]].format(
                molecular_charge=pref["molecular_charge"],
                nstep=pref["nstep"],
                multiplicity=pref["multiplicity"],
                basis=basis_dict[pref["basis"]],
                DFTBSK_part=DFTBSK,
                xyz_part=xyz_part))

    inp_file = "{}.inp".format(base_name)
    out_file = "{}.out".format(base_name)
    # Run GAMESS
    exe_cmd = "cd {}; {} {} {} {} {} > {}".format(hash_dir,
                                                  mpi_cmd,
                                                  pref["gamess_loc"],
                                                  inp_file,
                                                  pref["verno"],
                                                  str(pref["nproc"]),
                                                  out_file)
    print("Running GAMESS:\n", exe_cmd)
    single_cmd(exe_cmd)
    out_file = "{}/{}.out".format(hash_dir, base_name)
    inp_file = "{}/{}.inp".format(hash_dir, base_name)
    if pref["mode"] == "vib":
        xyz_list = pref["input"]
        try:
            eng = gamess2eng(out_file)
        except:
            eng = "nan"
        try:
            freq_list = gamess2frequency(out_file, atom_number)
        except:
            freq_list = []
        try:
            inten_list = gamess2intensity(out_file, atom_number)
        except:
            inten_list = []
    else:
        try:
            xyz_list = gamess2xyz(atom_number, dat_file)
        except:
            xyz_list = []
        try:
            eng = gamess2eng(out_file)
        except:
            eng = "nan"
        freq_list = []
        inten_list = []
    if not pref["preserve_inp"]:
        single_cmd("rm -rf {}".format(hash_dir))
    if(pref["debugging_mode"] == True):
        print('\n------------------------{}--------------------------\n'.format(base_name))
        #print("freq_list = ", freq_list, '\n')
        #print("inten_list = ", inten_list, '\n')
        #print("xyz_list = ", xyz_list, '\n')
        print("eng = ", eng, '\n')
    return result({"freq_list": freq_list,
                   "inten_list": inten_list,
                   "xyz_list": xyz_list,
                   "eng": eng}, pref)

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
            std_err("Error! pref={} is not a dict!".format(kwargs["pref"]))
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
        results: any (normal results)
        pref: preference dict
    Note:
        0 could be successful return value.
        1 or other none-zero values for fault results.
    """
    if pref["use_pref"]:
        pref["return"] = results
        return pref
    else:
        return results
# } pref_tools

def gamess2xyz(atom_number, dat_file):
    """
    Created by CJ Huang in 20180711
    Return:
        xyz_list: a list of every atoms' xyz coordinates.
    Args:
        dat_file: a string of a .dat file's name. (.dat is generated by GAMESS)
    Note:
        Extract xyz data from .dat file.
        head: line number of the head of the data we wanted.
        buttom: line number of the bottom of the data we wanted.
    """
    #extration of xyz data.
    extract_line_cmd = "grep -n 'NSERCH' {} | tail -n 1 | cut -d ' ' -f 1 | cut -d ':' -f 1".format(dat_file)
    data_wanted_head_line = single_cmd(extract_line_cmd)
    head   = int(data_wanted_head_line[0]) + 4
    buttom = head + atom_number - 1
    xyz_list = []
    for i in range(atom_number):
        line = head + i
        atom_list = []
        extract_line_cmd = "sed -n '{}, {}p' {} | awk '{}'".format(line,
                                                                   line,
                                                                   dat_file,
                                                                   "{print $1}")
        atom = single_cmd(extract_line_cmd)[0].rstrip("\n")
        if len(atom) == 2:
            atom = atom[0]+atom[1].lower()
        atom_list.append(atom)
        for j in range(3,6):
            extract_line_cmd = "sed -n '{}, {}p' {} | awk '{}{}{}{}'".format(line,
                                                                             line,
                                                                             dat_file,
                                                                             "{print ",
                                                                             "$",
                                                                             str(j),
                                                                             "}")
            atom_list.append(float(single_cmd(extract_line_cmd)[0]))
        xyz_list.append(atom_list)
    return xyz_list


def gamess2eng(out_file):
    """
    Created by CJ Huang in 20180711
    Return:
        eng: a float represents the energy after minimization.(E(RDFTB) in .dat file)
    Args:
        out_file: a string of a .out file's name. (.dat is generated by GAMESS)
    Note:
        Extract energy from .out file.
    """
    #extraction of energy E(REDFTB) data
    extract_cmd = "grep 'EXECUTION\ OF\ GAMESS\ TERMINATED\ -ABNORMALLY-' {}"\
.format(out_file)
    err_test = single_cmd(extract_cmd)
    if err_test:
        return 'nan'
    else:
        vib_test_cmd = "grep 'VIBRATION' {}".format(out_file)
        vib_test = single_cmd(vib_test_cmd)
        if vib_test:
            extract_cmd = "grep 'FINAL' {}".format(out_file)
            eng_line = single_cmd(extract_cmd)
            eng = float(eng_line[0].split('IS')[1].split('AFTER')[0])
        else:
            extract_cmd = "grep 'NSERCH=' {} | tail -n 1 | cut -d '=' -f 3".format(out_file)
            eng = float(single_cmd(extract_cmd)[0])
        return eng


def gamess2frequency(out_file, atom_number):
    """
    Created by CJ Huang in 20180718
    Return:
        freq_list: a list of every modes' frequencies.
    Args:
        out_file: a string of a .out file's name. (.dat is generated by GAMESS)
        atom_number: the total number of atoms
    Note:
        Extract frequency data from .dat file.
    """
    mode = 3*atom_number - 6
    extract_line_cmd = "grep 'THERMOCHEMISTRY\.' {} | cut -d '-' -f 2 | \
                        cut -d 'V' -f 1".format(out_file)
    modes = int(single_cmd(extract_line_cmd)[0])
    rt_mode_cmd = "grep 'ROTATIONS' {}".format(out_file)
    rt_line = single_cmd(rt_mode_cmd)[0]
    first_rt_mode = int(rt_line.split('TO')[0].split('MODES')[1])
    last_rt_mode  = int(rt_line.split('TO')[1].split('ARE')[0])
    imag_mode_number = first_rt_mode - 1
    extract_line_cmd = "grep -n 'ORTHONORMALIZED' {}| cut -d ':' -f 1".format(out_file)
    head = int(single_cmd(extract_line_cmd)[0]) + 4
    buttom = head + modes + last_rt_mode - 1
    if(imag_mode_number != 0):
        exe_cmd_imag = "cat {} | sed -n '{}, {}p' | awk '{}'".format(out_file,
                                                                     head,
                                                                     head+imag_mode_number-1,
                                                                     "{print $2}")
        tmp = single_cmd(exe_cmd_imag)
        imag_freq_list = []
        for item in tmp:
            imag_freq_list.append((-1.0) * float(item.rstrip("\n")))
    elif(imag_mode_number == 0):
        imag_freq_list = []
    print('\n-------------------------------------------------------\n')
    print('N = ', atom_number)
    print('The number of imaginary modes: ', len(imag_freq_list))
    if(len(imag_freq_list) != 0):
        print("\033[31mIMAGINARY MODE OCCUR!\033[0m")
    exe_cmd = "cat {} | sed -n '{}, {}p' | awk '{}'".format(out_file,
                                                            head+last_rt_mode,
                                                            buttom,
                                                            "{print $2}")
    tmp = single_cmd(exe_cmd)
    freq_list = []
    for item in tmp:
        try:
            freq_list.append(float(item.rstrip('\n')))
        except:
            break
    freq_list = freq_list + imag_freq_list
    print('The number of modes(Supposed to be 3*N-6 = {}) = : '.format(str(mode)), len(freq_list), '\n')
    if(mode == len(freq_list)):
        print("\033[32mModes number Correct!\033[0m")
    else:
        print("\033[31mError.\033[0m")
    return freq_list


def gamess2intensity(out_file, atom_number):
    """
    Created by CJ Huang in 20180718
    Return:
        freq_list: a list of every modes' frequencies.
    Args:
        out_file: a string of a .out file's name. (.dat is generated by GAMESS)
        atom_number: the total number of atoms
    Note:
        Extract frequency data from .dat file.
    """
    mode = 3*atom_number - 6
    extract_line_cmd = "grep 'THERMOCHEMISTRY\.' {} | cut -d '-' -f 2 | \
                        cut -d 'V' -f 1".format(out_file)
    modes = int(single_cmd(extract_line_cmd)[0])
    rt_mode_cmd = "grep 'ROTATIONS' {}".format(out_file)
    rt_line = single_cmd(rt_mode_cmd)[0]
    first_rt_mode = int(rt_line.split('TO')[0].split('MODES')[1])
    last_rt_mode  = int(rt_line.split('TO')[1].split('ARE')[0])
    imag_mode_number = first_rt_mode - 1
    extract_line_cmd = "grep -n 'ORTHONORMALIZED' {}| cut -d ':' -f 1".format(out_file)
    head = int(single_cmd(extract_line_cmd)[0]) + 4
    buttom = head + modes + last_rt_mode - 1
    if(imag_mode_number != 0):
        exe_cmd_imag = "cat {} | sed -n '{}, {}p' | awk '{}'".format(out_file,
                                                                     head,
                                                                     head+imag_mode_number-1,
                                                                     "{print $5}")
        tmp = single_cmd(exe_cmd_imag)
        imag_inten_list = []
        for item in tmp:
            imag_inten_list.append(float(item.rstrip("\n")))
    elif(imag_mode_number == 0):
        imag_inten_list = []
    exe_cmd = "cat {} | sed -n '{}, {}p' | awk '{}'".format(out_file,
                                                            head+last_rt_mode,
                                                            buttom,
                                                            "{print $5}")
    tmp = single_cmd(exe_cmd)
    inten_list = []
    for item in tmp:
        try:
            inten_list.append(float(item.rstrip("\n")))
        except:
            break
    inten_list = inten_list + imag_inten_list
    return inten_list


def main():
    """ main """
    test = False
    if(test):
        #modify test to run example.
        #------------------------------EXAMPLE----------------------------------#
        par = {"mode" : "vib",                                                  #
               "input": "/home/cjhuang/test2.xyz",                              #
               "molecular_charge" : "1",                                        #
               "parm_loc" : "/home/cjhuang/src/gamess/parameters",              #
               "gamess_loc" : "/home/cjhuang/src/gamess/rungms",                #
               "debugging_mode" : True,                                         #
               "preserve_inp" : True,                                           #
               "verno" : "00",                                                  #
               "tmp": "/tmp",                                                   #
               }                                                                #
        gamess_vibmin(pref=par)                                                 #
        #-----------------------------------------------------------------------#
if __name__ == "__main__":
    main()
