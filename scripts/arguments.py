import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-name','-n',required=True,nargs='+')
parser.add_argument('-HSTband','-b',default='nuv')
parser.add_argument('-scalepc','-s',default=32)
parser.add_argument('-version','-v',default='v1p2')

args = parser.parse_args()
name = args.name
version = args.version
HSTband = args.HSTband
scalepc = args.scalepc

def main(name,version,HSTband,scalepc):

    if name == 4:
        continue

    print(f'{name} {version} {HSTband} {scalepc}')

if __name__ == '__main__':

    for i in range(5):
        main(i,version,HSTband,scalepc)