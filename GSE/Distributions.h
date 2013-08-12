#ifndef GSE_DISTRIBUTIONS_H
#define GSE_DISTRIBUTIONS_H

#include "Gaussian.h"
#include "DiracDistribution.h"
//#include "ParticleDistribution.h"

GSE_NS_BEGIN

template<typename Dist>
DistributionSampler<Dist> make_distSampler(const Dist& d) { return {d}; };

GSE_NS_END

#endif
