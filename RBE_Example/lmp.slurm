#!/bin/bash

#SBATCH -J lammps 
#SBATCH -p 64c512g
#SBATCH -N 1   
#SBATCH --ntasks-per-node=64 
#SBATCH --mail-type=end
#SBATCH --mail-user=gaowh2019@sjtu.edu.cn
#SBATCH -o RBE.out
#SBATCH -e %j.err  
#SBATCH --time=80:00:00
  
#SBATCH --exclusive  


module purge
#module load mpich fftw
#module load intel-parallel-studio
#module load lammps/20190807-intel-19.0.5-impi 

module load oneapi
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi  
  
ulimit -s unlimited 
ulimit -l unlimited 

#export OMP_NUM_THREADS=40

for i in 80          
do

#srun --mpi=pmi2 /lustre/home/acct-matxzl/matxzl/jiuyang/InvAug/src/lmp_intel_cpu_intelmpi -i salt.in  

#mv Time.txt Time_RBEwald_${i}_17789.txt
#mv Time_Real_Intel.txt Time_Real_RBEwald_${i}_17789.txt      

#srun --mpi=pmi2 /lustre/home/acct-matxzl/matxzl/jiuyang/InvAug/src/lmp_intel_cpu_intelmpi -i salt_pppm.in 

#mv Time_PPPM.txt Time_PPPM_${i}_1e-4_17789.txt
#mv Time_Real.txt Time_Real_PPPM_${i}_1e-4_17789.txt      

srun --mpi=pmi2 -n 4 ../src/lmp_intel_cpu_intelmpi -i in.spce-bulk-nvt 

done

