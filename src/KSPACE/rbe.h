#ifdef KSPACE_CLASS

KSpaceStyle(rbe,Rbe)

#else

#ifndef LMP_RBE_H
#define LMP_RBE_H

#include "kspace.h"
#include<iostream>
#include <fstream>
#include <sstream>

namespace LAMMPS_NS {
	
class Rbe : public KSpace {
public:
  Rbe(class LAMMPS *);
  virtual ~Rbe();
  void settings(int, char**) override;
  void init() override;
  void setup() override;
  void compute(int, int) override;
  double memory_usage() override;
  void slabcorr();

protected:
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

