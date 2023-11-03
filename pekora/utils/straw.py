import os
import subprocess as sp

from .. import const

def run_straw(
    chr1_region:str,
    resolution:int,
    input_fpath:str,
    output_f:str,
    balancing:str="NONE",
    counts:str="observed",
    bp_frag:str="BP",
    chr2_region:str=None,
    overwrite:bool=False,
    close_output:bool=True
):
    
    assert counts in [e.value for e in const.Counts], f"Invalid data: {counts}"
    assert balancing in [e.value for e in const.Balancing], f"Invalid normalization: {balancing}"
    assert bp_frag in [e.value for e in const.BpFrag], f"Invalid BP/FRAG: {bp_frag}"
    assert type(chr1_region) is str, f"chr1_region must be string"
    
    #? in-memory mode
    if output_f is None:
        in_memory = True
        f = sp.PIPE 
    elif type(output_f) == str:
        in_memory = False
        assert not os.path.exists(output_f) or overwrite, "File exist!"
        f = open(output_f, "w")
    else:
        in_memory = False
        f = output_f
        
    args = [
        "straw",
        counts,
        balancing,
        input_fpath,
        chr1_region,
        chr1_region if chr2_region is None else chr2_region,
        bp_frag,
        str(int(resolution)),
    ]
    
    res = sp.run(
        args, 
        stderr=sp.PIPE, 
        stdout=f
    )
    
    assert res.returncode == 0, res.stderr.decode("utf-8")
    
    if close_output and not in_memory:
        f.close()
        
    if in_memory:
        return res.stdout