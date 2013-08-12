#ifndef GSE_DIRAC_DISTRIBUTION_H
#define GSE_DIRAC_DISTRIBUTION_H

#include"GSEDefs.h"
#include "ManifoldUtil.h"
#include <Eigen/Core>

GSE_NS_BEGIN

using namespace Eigen;

//-----------------------------------------------------------------------------

template<typename Manifold_>
struct DiracDistribution
{
	typedef Manifold_ Manifold;
	Matrix<double,Manifold::Dim,1> mu;
};

template<typename Dist>
class DistributionSampler;

template<typename Manifold>
class DistributionSampler<DiracDistribution<Manifold> >
{
	DiracDistribution<Manifold> d;
public:
	DistributionSampler<DiracDistribution<Manifold> >(const DiracDistribution<Manifold>& d_) : d(d_) {}
	typename Manifold::Vec sample()
	{
		return d.mu;
	}
};

template<int N>
DiracDistribution<EuclideanManifold<N> > make_dirac(const Matrix<double,N,1>& d) { return {d}; }

GSE_NS_END

#endif
