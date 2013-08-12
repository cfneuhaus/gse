#ifndef GSE_UKF_SQRTGAUSSIAN_H
#define GSE_UKF_SQRTGAUSSIAN_H

#include"GSEDefs.h"
#include <Eigen/Dense>
#include <math.h>

GSE_NS_BEGIN

//-----------------------------------------------------------------------------

template<typename Manifold_>
struct SqrtGaussian
{
	typedef Manifold_ Manifold;
	typedef Matrix<double,Manifold::Dim,1> MuType;
	typedef Matrix<double,Manifold::DeltaDim,Manifold::DeltaDim> CovType;

	MuType mu;
	CovType covL;

	SqrtGaussian() {}
	SqrtGaussian(const MuType& mu_, const CovType& covL_) : mu(mu_), covL(covL_) {}
	explicit SqrtGaussian(const CovType& covL_) : covL(covL_) { mu.setZero(); }
	explicit SqrtGaussian(const Gaussian<Manifold>& g) : mu(g.mu)
	{
		covL=g.cov.llt().matrixL();
	}

	Eigen::TriangularView<CovType,Eigen::Lower> getCovL() const { return covL; }

	Gaussian<Manifold> toGaussian() const { return {mu,covL*covL.transpose()}; }

	double sqrMahalanobisDist(const typename Manifold::Vec& x) const
	{
		typename Manifold::DeltaVec d=Manifold::manifoldSub(x,mu);
		getCovL().solveInPlace(d);
		return d.transpose()*d;
	}
	double densityAt(const typename Manifold::Vec& x) const
	{
		return 1.0/(pow(2*M_PI,Manifold::DeltaDim/2.0)*covL.determinant())*exp(-0.5*sqrMahalanobisDist(x));
	}

	template<int K>
	void addDelta(const typename Manifold::DeltaVec& delta, const Matrix<double,Manifold::DeltaDim,K>& U)
	{
		mu=Manifold::manifoldAdd(mu,delta);

		// not sure if this really requires a loop...
		for (int i=0;i<U.cols();i++)
		{
			if (internal::llt_inplace<double,Lower>::rankUpdate(covL,U.col(i),-1)>=0)
			{
				//std::cout << "Numerical Issue in RankUpdate2" << std::endl;
			}
		}
	}
};

template<int N>
SqrtGaussian<EuclideanManifold<N> > make_euclideanSqrtGaussian(const Matrix<double,N,1>& mu, const Matrix<double,N,N>& covL) { return {mu,covL}; }
template<int N>
SqrtGaussian<EuclideanManifold<N> > make_euclideanSqrtGaussian(const Matrix<double,N,N>& covL) { return {covL}; }
template<typename Manifold>
SqrtGaussian<Manifold> make_sqrtGaussian(const Gaussian<Manifold>& g) { return SqrtGaussian<Manifold>{g}; }

template<typename M1, typename M2>
SqrtGaussian<ProductManifold<M1,M2> > appendGaussians(const SqrtGaussian<M1>& g1, const SqrtGaussian<M2>& g2)
{
	SqrtGaussian<ProductManifold<M1,M2> > ret;
	ret.mu << g1.mu,g2.mu;
	ret.covL.setZero();
	ret.covL.template block<M1::DeltaDim,M1::DeltaDim>(0,0)=g1.covL;
	ret.covL.template block<M2::DeltaDim,M2::DeltaDim>(M1::DeltaDim,M1::DeltaDim)=g2.covL;
	return ret;
}

//-----------------------------------------------------------------------------


GSE_NS_END

#endif
