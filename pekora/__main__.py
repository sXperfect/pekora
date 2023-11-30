import io
import argparse
import tempfile as temp
from . import const
from . import mains

CHR1_REG_HELP = "Region of <chr1>"
CHR2_REG_HELP = "Region of <chr2>"
RES_HELP = "Resolution"
BALANCING_HELP = 'Name of balancing method'
INPUT_HELP = "Input file path"
POINTS_HELP = "Points file path"
OUTPUT_HELP = "Output file path"

parser = argparse.ArgumentParser(description="", prog="")
parser.add_argument('-l', '--log_level', help='log level', choices=const.LOG_LEVELS.keys(), default='error')
parser.add_argument('--overwrite', action='store_true')
subparsers = parser.add_subparsers(dest='mode')

create_h3dg_coo_parser = subparsers.add_parser('create_h3dg_coo')
create_h3dg_coo_parser.add_argument('chr_region', metavar='chr_region', help=CHR1_REG_HELP)
create_h3dg_coo_parser.add_argument('resolution', metavar='resolution', help=RES_HELP, type=int)
create_h3dg_coo_parser.add_argument('balancing', metavar='balancing', help=BALANCING_HELP, choices=["KR", "VC", "VC_SQRT"])
create_h3dg_coo_parser.add_argument('input', metavar='input_file_path', help=INPUT_HELP)
create_h3dg_coo_parser.add_argument('output', metavar='output_file_path', help=OUTPUT_HELP)

create_contact_coo_parser = subparsers.add_parser('create_contact_coo')
create_contact_coo_parser.add_argument('--chr2_region', metavar='chr2_region', help=CHR2_REG_HELP)
create_contact_coo_parser.add_argument('--output-resolution-one', action='store_true')
create_contact_coo_parser.add_argument('--output-delimiter', default=const.DEF_SEP, type=str)
create_contact_coo_parser.add_argument('--normalize-pos', action='store_true')
create_contact_coo_parser.add_argument('--gen-pseudo-weights', action='store_true')
create_contact_coo_parser.add_argument('chr1_region', metavar='chr1_region', help=CHR1_REG_HELP)
create_contact_coo_parser.add_argument('resolution', metavar='resolution', help=RES_HELP, type=int)
create_contact_coo_parser.add_argument('balancing', metavar='balancing', help=BALANCING_HELP, choices=const.AVAIL_BALANCINGS)
create_contact_coo_parser.add_argument('input', metavar='input_file_path', help=INPUT_HELP)
create_contact_coo_parser.add_argument('output', metavar='output_file_path', help=OUTPUT_HELP)

create_contact_mcoo_parser = subparsers.add_parser('create_contact_mcoo')
create_contact_mcoo_parser.add_argument('--chr2_region', metavar='chr2_region', help=CHR2_REG_HELP)
create_contact_mcoo_parser.add_argument('--output-resolution-one', action='store_true')
create_contact_mcoo_parser.add_argument('chr1_region', metavar='chr1_region', help=CHR1_REG_HELP)
create_contact_mcoo_parser.add_argument('resolution', metavar='resolution', help=RES_HELP, type=int)
create_contact_mcoo_parser.add_argument('balancing', metavar='balancing', help=BALANCING_HELP, choices=const.AVAIL_BALANCINGS)
create_contact_mcoo_parser.add_argument('input', metavar='input_file_path', help=INPUT_HELP)
create_contact_mcoo_parser.add_argument('output', metavar='output_file_path', help=OUTPUT_HELP)

comp_superrec_spearmanr_parser = subparsers.add_parser('comp_superrec_spearmanr')
comp_superrec_spearmanr_parser.add_argument('--chr2_region', metavar='chr2_region', help=CHR2_REG_HELP)
comp_superrec_spearmanr_parser.add_argument('chr1_region', metavar='chr1_region', help=CHR1_REG_HELP)
comp_superrec_spearmanr_parser.add_argument('resolution', metavar='resolution', help=RES_HELP, type=int)
comp_superrec_spearmanr_parser.add_argument('balancing', metavar='balancing', help=BALANCING_HELP, choices=const.AVAIL_BALANCINGS)
comp_superrec_spearmanr_parser.add_argument('input', metavar='data_file_path', help=INPUT_HELP)
comp_superrec_spearmanr_parser.add_argument('points', metavar='points_file_path', help=POINTS_HELP)

comp_shneigh_spearmanr_parser = subparsers.add_parser('comp_shneigh_spearmanr')
comp_shneigh_spearmanr_parser.add_argument('--chr2_region', metavar='chr2_region', help=CHR2_REG_HELP)
comp_shneigh_spearmanr_parser.add_argument('chr1_region', metavar='chr1_region', help=CHR1_REG_HELP)
comp_shneigh_spearmanr_parser.add_argument('resolution', metavar='resolution', help=RES_HELP, type=int)
comp_shneigh_spearmanr_parser.add_argument('balancing', metavar='balancing', help=BALANCING_HELP, choices=const.AVAIL_BALANCINGS)
comp_shneigh_spearmanr_parser.add_argument('input', metavar='data_file_path', help=INPUT_HELP)
comp_shneigh_spearmanr_parser.add_argument('points', metavar='points_file_path', help=POINTS_HELP)

comp_h3dg_spearmanr_parser = subparsers.add_parser('comp_h3dg_spearmanr')
comp_h3dg_spearmanr_parser.add_argument('--chr2_region', metavar='chr2_region', help=CHR2_REG_HELP)
comp_h3dg_spearmanr_parser.add_argument('chr1_region', metavar='chr1_region', help=CHR1_REG_HELP)
comp_h3dg_spearmanr_parser.add_argument('resolution', metavar='resolution', help=RES_HELP, type=int)
comp_h3dg_spearmanr_parser.add_argument('balancing', metavar='balancing', help=BALANCING_HELP, choices=const.AVAIL_BALANCINGS)
comp_h3dg_spearmanr_parser.add_argument('input', metavar='data_file_path', help=INPUT_HELP)
comp_h3dg_spearmanr_parser.add_argument('points', metavar='points_file_path', help=POINTS_HELP)
comp_h3dg_spearmanr_parser.add_argument('mapping', metavar='mapping_file_path', help="")

comp_flamingo_spearmanr_parser = subparsers.add_parser('comp_flamingo_spearmanr')
comp_flamingo_spearmanr_parser.add_argument('--chr2_region', metavar='chr2_region', help=CHR2_REG_HELP)
comp_flamingo_spearmanr_parser.add_argument('chr1_region', metavar='chr1_region', help=CHR1_REG_HELP)
comp_flamingo_spearmanr_parser.add_argument('resolution', metavar='resolution', help=RES_HELP, type=int)
comp_flamingo_spearmanr_parser.add_argument('balancing', metavar='balancing', help=BALANCING_HELP, choices=const.AVAIL_BALANCINGS)
comp_flamingo_spearmanr_parser.add_argument('input', metavar='data_file_path', help=INPUT_HELP)
comp_flamingo_spearmanr_parser.add_argument('points', metavar='points_file_path', help=POINTS_HELP)

comp_myflamingo_spearmanr_parser = subparsers.add_parser('comp_myflamingo_spearmanr')
comp_myflamingo_spearmanr_parser.add_argument('--chr2_region', metavar='chr2_region', help=CHR2_REG_HELP)
comp_myflamingo_spearmanr_parser.add_argument('chr1_region', metavar='chr1_region', help=CHR1_REG_HELP)
comp_myflamingo_spearmanr_parser.add_argument('resolution', metavar='resolution', help=RES_HELP, type=int)
comp_myflamingo_spearmanr_parser.add_argument('balancing', metavar='balancing', help=BALANCING_HELP, choices=const.AVAIL_BALANCINGS)
comp_myflamingo_spearmanr_parser.add_argument('input', metavar='data_file_path', help=INPUT_HELP)
comp_myflamingo_spearmanr_parser.add_argument('points', metavar='points_file_path', help=POINTS_HELP)

comp_pekora_spearmanr_parser = subparsers.add_parser('comp_pekora_spearmanr')
comp_pekora_spearmanr_parser.add_argument('--chr2_region', metavar='chr2_region', help=CHR2_REG_HELP)
comp_pekora_spearmanr_parser.add_argument('chr1_region', metavar='chr1_region', help=CHR1_REG_HELP)
comp_pekora_spearmanr_parser.add_argument('resolution', metavar='resolution', help=RES_HELP, type=int)
comp_pekora_spearmanr_parser.add_argument('balancing', metavar='balancing', help=BALANCING_HELP, choices=const.AVAIL_BALANCINGS)
comp_pekora_spearmanr_parser.add_argument('input', metavar='data_file_path', help=INPUT_HELP)
comp_pekora_spearmanr_parser.add_argument('points', metavar='points_file_path', help=POINTS_HELP)

def run(args):
    if args.mode == 'create_h3dg_coo':
        delimiter = " "
        
        #? Generate raw counts
        mains.create_contact_coo(
            args.chr_region,
            args.resolution,
            "NONE",
            args.input,
            f"{args.output}.raw",
            norm_pos=False,
            generate_pseudo_weights=False,
            output_delimiter=delimiter,
        )
        
        #? Generate normalized counts
        mains.create_contact_coo(
            args.chr_region,
            args.resolution,
            args.balancing,
            args.input,
            f"{args.output}.norm",
            norm_pos=False,
            generate_pseudo_weights=False,
            output_delimiter=delimiter,
        )
        
    elif args.mode == 'create_contact_coo':
        mains.create_contact_coo(
            args.chr1_region,
            args.resolution,
            args.balancing,
            args.input,
            args.output,
            chr2_region=args.chr2_region,
            res_to_one=args.output_resolution_one,
            norm_pos=args.normalize_pos,
            generate_pseudo_weights=args.gen_pseudo_weights,
            output_delimiter=args.output_delimiter
        )
    elif args.mode == 'create_contact_mcoo':
        mains.create_contact_mcoo(
            args.chr1_region,
            args.resolution,
            args.balancing,
            args.input,
            args.output,
            chr2_region=args.chr2_region,
        )
    elif args.mode == 'comp_superrec_spearmanr':
        mains.comp_superrec_spearmanr(
            args.chr1_region,
            args.resolution,
            args.balancing,
            args.input,
            args.points,
        )
    elif args.mode == 'comp_shneigh_spearmanr':
        mains.comp_shneigh_spearmanr(
            args.chr1_region,
            args.resolution,
            args.balancing,
            args.input,
            args.points,
        )
    elif args.mode == 'comp_h3dg_spearmanr':
        mains.comp_h3dg_spearmanr(
            args.chr1_region,
            args.resolution,
            args.balancing,
            args.input,
            args.points,
            args.mapping,
        )
    elif args.mode == 'comp_flamingo_spearmanr':
        mains.comp_flamingo_spearmanr(
            args.chr1_region,
            args.resolution,
            args.balancing,
            args.input,
            args.points,
        )
    elif args.mode == 'comp_myflamingo_spearmanr':
        mains.comp_myflamingo_spearmanr(
            args.chr1_region,
            args.resolution,
            args.balancing,
            args.input,
            args.points,
        )
    elif args.mode == 'comp_pekora_spearmanr':
        mains.comp_pekora_spearmanr(
            args.chr1_region,
            args.resolution,
            args.balancing,
            args.input,
            args.points,
        )
    else:
        raise ValueError(f"Mode not supported:{args.mode}")
    
    
args = parser.parse_args()
run(args)