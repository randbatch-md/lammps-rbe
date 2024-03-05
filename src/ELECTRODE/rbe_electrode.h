/* ----------------------------------------------------------------------
   Contributing authors: Weihang Gao(SJTU)
------------------------------------------------------------------------- */
#ifdef KSPACE_CLASS

KSpaceStyle(rbe/electrode,RbeElectrode)

#else

#ifndef LMP_RBE_ELECTRODE_H
#define LMP_RBE_ELECTRODE_H

#include "electrode_kspace.h"
#include "rbe.h"
#include<iostream>
#include <fstream>
#include <sstream>

namespace LAMMPS_NS {
	
class RbeElectrode : public Rbe, public ElectrodeKSpace {
public:
  RbeElectrode(class LAMMPS *);
  virtual ~RbeElectrode();
  void settings(int, char**) override;
  void init() override;
  void setup() override;
  void compute(int, int) override;
  double memory_usage() override;
  void prepareSampleK();
  //void slabcorr();

  // k-space part of coulomb matrix computation
  void compute_vector(double *, int, int, bool) override;
  void compute_vector_corr(double *, int, int, bool) override;
  void compute_matrix(bigint *, double **, bool) override;
  void compute_matrix_corr(bigint *, double **) override;

protected:
class BoundaryCorrection *boundcorr;
  double alpha, alpha_sqrt; //
  int P;                    //
  double S;
  double pi,pi_sqrt;
  double volume;
  std::streamoff pointer; 
  std::ifstream infile;
  int lx,ly,lz;
  int Step;
  int me;
  int (*K_All)[3];
  float *Time;
  int RankID; //
  double S1_X, S1_Y, S1_Z;
  float** K_Sample;
};
}
#endif
#endif

