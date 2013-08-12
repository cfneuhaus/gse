#ifndef GSE_UKF_GAUSSIAN_H
#define GSE_UKF_GAUSSIAN_H

#include"GSEDefs.h"
#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/normal_distribution.hpp>
#include <math.h>

GSE_NS_BEGIN

template<typename Dist>
class DistributionSampler;

//-----------------------------------------------------------------------------

template<typename Manifold_>
struct Gaussian
{
	typedef Manifold_ Manifold;
	typedef Matrix<double,Manifold::Dim,1> MuType;
	typedef Matrix<double,Manifold::DeltaDim,Manifold::DeltaDim> CovType;

	Gaussian(const MuType& mu_, const CovType& cov_) : mu(mu_), cov(cov_) {}
	Gaussian(const CovType& cov_) : cov(cov_) { mu.setZero(); }

	MuType mu;
	CovType cov;

	double sqrMahalanobisDist(const typename Manifold::Vec& x) const
	{
		typename Manifold::DeltaVec d=Manifold::manifoldSub(x,mu);
		typename Manifold::DeltaVec d2=d;
		cov.llt().solveInPlace(d2);
		return d.transpose()*d2;
	}
	double densityAt(const typename Manifold::Vec& x) const
	{
		return 1.0/(pow(2*M_PI,Manifold::DeltaDim/2.0)*sqrt(cov.determinant()))*exp(-0.5*sqrMahalanobisDist(x));
	}

	template<int K>
	void addDelta(const typename Manifold::DeltaVec& delta, const Matrix<double,Manifold::DeltaDim,K>& U)
	{
		mu=Manifold::manifoldAdd(mu,delta);
		cov-=U*U.transpose();
	}
};

template<int N>
Gaussian<EuclideanManifold<N> > make_euclideanGaussian(const Matrix<double,N,1>& mu, const Matrix<double,N,N>& cov) { return {mu,cov}; };
template<int N>
Gaussian<EuclideanManifold<N> > make_euclideanGaussian(const Matrix<double,N,N>& cov) { return {cov}; };

template<typename _Scalar, int _size>
class EigenMultivariateNormal
{
	boost::mt19937 rng;    // The uniform pseudo-random algorithm
	boost::normal_distribution<_Scalar> norm;  // The gaussian combinator
	boost::variate_generator<boost::mt19937&,boost::normal_distribution<_Scalar> >
	randN; // The 0-mean unit-variance normal generator

	Eigen::Matrix<_Scalar,_size,_size> rot;
	Eigen::Matrix<_Scalar,_size,1> scl;

	Eigen::Matrix<_Scalar,_size,1> mean;

public:
	EigenMultivariateNormal(const Eigen::Matrix<_Scalar,_size,1>& meanVec,
							const Eigen::Matrix<_Scalar,_size,_size>& covarMat)
		: randN(rng,norm)
	{
		setCovar(covarMat);
		setMean(meanVec);
	}

	void setCovar(const Eigen::Matrix<_Scalar,_size,_size>& covarMat)
	{
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix<_Scalar,_size,_size> >
				eigenSolver(covarMat);
		rot = eigenSolver.eigenvectors();
		scl = eigenSolver.eigenvalues();
		for (int ii=0;ii<_size;++ii) {
			scl(ii,0) = sqrt(scl(ii,0));
		}
	}

	void setMean(const Eigen::Matrix<_Scalar,_size,1>& meanVec)
	{
		mean = meanVec;
	}

	void nextSample(Eigen::Matrix<_Scalar,_size,1>& sampleVec)
	{
		for (int ii=0;ii<_size;++ii) {
			sampleVec(ii,0) = randN()*scl(ii,0);
		}
		sampleVec = rot*sampleVec + mean;
	}

};

template<typename Manifold>
class DistributionSampler<Gaussian<Manifold> >
{
	Gaussian<Manifold> g;
	EigenMultivariateNormal<double,Manifold::DeltaDim> emv;
public:
	DistributionSampler<Gaussian<Manifold> >(const Gaussian<Manifold>& g_) : g(g_), emv(Manifold::DeltaVec::Zero(),g.cov)
	{
	}
	typename Manifold::Vec sample()
	{
		typename Manifold::DeltaVec s;
		emv.nextSample(s);
		return Manifold::manifoldAdd(g.mu,s);
	}
};

//-----------------------------------------------------------------------------

GSE_NS_END

#endif
