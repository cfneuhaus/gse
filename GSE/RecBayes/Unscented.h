#ifndef GSE_RECBAYES_UNSCENTED_H
#define GSE_RECBAYES_UNSCENTED_H

#include "../GSEDefs.h"
#include <Eigen/Core>
#include <Eigen/QR>
#include "../ManifoldUtil.h"
#include "../Gaussian.h"
#include "../SqrtGaussian.h"

GSE_NS_BEGIN

using namespace Eigen;

//-----------------------------------------------------------------------------

namespace Unscented
{
	template<typename Manifold>
	struct StandardScheme
	{
		static const int NumSigmaPoints=2*Manifold::DeltaDim+1;

		double lambda;
		double s0;
		double si;
		double c0;
		double ci;
		StandardScheme(double alpha=0.001, double beta=2, double kappa=0)
		{
			lambda=alpha*alpha*(Manifold::DeltaDim+kappa)-Manifold::DeltaDim;
			s0=lambda/(Manifold::DeltaDim+lambda);
			si=1.0/(2*(Manifold::DeltaDim+lambda));
			c0=lambda/(Manifold::DeltaDim+lambda)+(1-alpha*alpha+beta);
			ci=1.0/(2*(Manifold::DeltaDim+lambda));
		}

		double unscentedStateWeight(int i) const
		{
			if (i==0)
				return s0;
			return si;
		}
		double unscentedCovWeight(int i) const
		{
			if (i==0)
				return c0;
			return ci;
		}


		Matrix<double,Manifold::Dim,NumSigmaPoints> unscentedSqrtTransform(const Matrix<double,Manifold::Dim,1>& x, const Matrix<double,Manifold::DeltaDim,Manifold::DeltaDim>& sqrtP) const
		{
			//assert(Manifold::Dim==Dim);

			const int L=Manifold::DeltaDim;
			Matrix<double,Manifold::Dim,NumSigmaPoints> ret;

			int curChi=0;

			ret.col(curChi++)=x;
			for (int sign=-1; sign<=1; sign+=2)
			{
				for (int i=0; i<L; i++)
					ret.col(curChi++)=Manifold::manifoldAdd(x,sign*sqrtf(L+lambda)*sqrtP.col(i));
			}
			return ret;
		}
		Matrix<double,Manifold::Dim,NumSigmaPoints> unscentedSqrtTransform(const SqrtGaussian<Manifold>& g) const
		{
			//assert(Manifold::Dim==Dim);

			const int L=Manifold::DeltaDim;
			Matrix<double,Manifold::Dim,NumSigmaPoints> ret;

			int curChi=0;

			ret.col(curChi++)=g.mu;
			for (int sign=-1; sign<=1; sign+=2)
			{
				for (int i=0; i<L; i++)
					ret.col(curChi++)=Manifold::manifoldAdd(g.mu,sign*sqrtf(L+lambda)*g.covL.col(i));
			}
			return ret;
		}
		Matrix<double,Manifold::Dim,NumSigmaPoints> unscentedTransform(const Matrix<double,Manifold::Dim,1>& x, const Matrix<double,Manifold::DeltaDim,Manifold::DeltaDim>& P) const
		{
			return unscentedSqrtTransform(x,P.llt().matrixL());
		}

		template<typename PtManifold>
		void unscentedSqrtCombineManifold(const Matrix<double,PtManifold::Dim,NumSigmaPoints>& vecs, Matrix<double,PtManifold::Dim,1>& mu, Matrix<double,PtManifold::DeltaDim,PtManifold::DeltaDim>& sqcov)
		{
			const int DeltaDim=PtManifold::DeltaDim;

			mu=vecs.col(0);
			for (int it=0;it<20;it++)
			{
				typename PtManifold::DeltaVec delta=PtManifold::DeltaVec::Zero();
				for (int i=0;i<NumSigmaPoints;i++)
					delta+=unscentedStateWeight(i)*PtManifold::manifoldSub(vecs.col(i),mu);
				mu=PtManifold::manifoldAdd(mu,delta);
				if (delta.norm()<0.00001)
					break;
			}

			Matrix<double,DeltaDim,NumSigmaPoints-1> tmp;
			for (int i=1; i<NumSigmaPoints; i++)
				tmp.col(i-1)=sqrt(unscentedCovWeight(i))*PtManifold::manifoldSub(vecs.col(i),mu);

			Matrix<double,NumSigmaPoints-1,DeltaDim> tmpT=tmp.transpose();

			Matrix<double,NumSigmaPoints-1,DeltaDim> R=tmpT.householderQr().householderQ().transpose()*tmpT;
			// R contains upper triangular part of cholesky decomp of tmpT^T*tmpT
			Matrix<double,DeltaDim,DeltaDim> Rpart=R.template block<DeltaDim,DeltaDim>(0,0);

			if (internal::llt_inplace<double,Upper>::rankUpdate(Rpart,PtManifold::manifoldSub(vecs.col(0),mu),unscentedCovWeight(0))>=0)
			{
				//std::cout << "Numerical Issue in RankUpdate" << std::endl;
			}

			sqcov=Rpart.transpose();
		}

		template<typename PtManifold>
		void unscentedSqrtCombineManifold(const Matrix<double,PtManifold::Dim,NumSigmaPoints>& vecs, SqrtGaussian<PtManifold>& g)
		{
			const int DeltaDim=PtManifold::DeltaDim;

			g.mu=vecs.col(0);
			for (int it=0;it<20;it++)
			{
				typename PtManifold::DeltaVec delta=PtManifold::DeltaVec::Zero();
				for (int i=0;i<NumSigmaPoints;i++)
					delta+=unscentedStateWeight(i)*PtManifold::manifoldSub(vecs.col(i),g.mu);
				g.mu=PtManifold::manifoldAdd(g.mu,delta);
				if (delta.norm()<0.00001)
					break;
			}

			Matrix<double,DeltaDim,NumSigmaPoints-1> tmp;
			for (int i=1; i<NumSigmaPoints; i++)
				tmp.col(i-1)=sqrt(unscentedCovWeight(i))*PtManifold::manifoldSub(vecs.col(i),g.mu);

			Matrix<double,NumSigmaPoints-1,DeltaDim> tmpT=tmp.transpose();

			Matrix<double,NumSigmaPoints-1,DeltaDim> R=tmpT.householderQr().householderQ().transpose()*tmpT;
			// R contains upper triangular part of cholesky decomp of tmpT^T*tmpT
			Matrix<double,DeltaDim,DeltaDim> Rpart=R.template block<DeltaDim,DeltaDim>(0,0);

			if (internal::llt_inplace<double,Upper>::rankUpdate(Rpart,PtManifold::manifoldSub(vecs.col(0),g.mu),unscentedCovWeight(0))>=0)
			{
				//std::cout << "Numerical Issue in RankUpdate" << std::endl;
			}

			g.covL=Rpart.transpose();
		}
	};
}
#if 0
//-----------------------------------------------------------------------------
template<int N, int M>
void Unscented::unscentedCombine(const Matrix<double,N,M>& vecs, Gaussian<N>& dest)
{
	const int L=(vecs.cols()-1)/2;
	//assert(int(vecs.cols())==2*L+1);

	dest.mu=Matrix<double,N,1>::Zero(vecs.rows());
	for (int i=0; i<2*L+1; i++)
		dest.mu+=unscentedStateWeight(L,i)*vecs.col(i);

	dest.cov=Matrix<double,N,N>::Zero(dest.mu.rows(), dest.mu.rows());
	for (int i=0; i<2*L+1; i++)
	{
		Matrix<double,N,1> chi_ineuzeroed=vecs.col(i)-dest.mu;
		dest.cov+=(unscentedCovWeight(L,i)*chi_ineuzeroed)*chi_ineuzeroed.transpose();
	}
}
//-----------------------------------------------------------------------------
#endif


template<typename Manifold_, typename AugManifold_, typename WeightingScheme_=Unscented::StandardScheme<ProductManifold<Manifold_,AugManifold_> > >
class AugmentedUnscentedDistribution
{
public:
	typedef Manifold_ Manifold;
	typedef AugManifold_ AugManifold;
	//typedef ProductManifold<Manifold,AugManifold> FullManifold;
	typedef typename boost::mpl::if_c<AugManifold::Dim==0,Manifold,ProductManifold<Manifold,AugManifold> >::type FullManifold;
	typedef WeightingScheme_ WeightingScheme;

	AugmentedUnscentedDistribution() {}
	AugmentedUnscentedDistribution(const SqrtGaussian<FullManifold>& g)
	{
		sqrtG.mu=g.mu.template segment<Manifold::Dim>(0);
		sqrtG.covL=g.covL.template block<Manifold::DeltaDim,Manifold::DeltaDim>(0,0);

		auto fullsigmapts=uscheme.unscentedSqrtTransform(g);
		sigmapts=fullsigmapts.template block<Manifold::Dim,WeightingScheme::NumSigmaPoints>(0,0);
		augsigmapts=fullsigmapts.template block<AugManifold::Dim,WeightingScheme::NumSigmaPoints>(Manifold::Dim,0);
	}
	AugmentedUnscentedDistribution(const SqrtGaussian<Manifold>& g,const SqrtGaussian<AugManifold>& gaug)
	{
		sqrtG=g;

		SqrtGaussian<FullManifold> fullg;
		fullg.mu << g.mu, gaug.mu;
		fullg.covL.setZero();
		fullg.covL.template block<Manifold::DeltaDim,Manifold::DeltaDim>(0,0)=g.covL;
		fullg.covL.template block<AugManifold::DeltaDim,AugManifold::DeltaDim>(Manifold::DeltaDim,Manifold::DeltaDim)=gaug.covL;

		auto fullsigmapts=uscheme.unscentedSqrtTransform(fullg);
		sigmapts=fullsigmapts.template block<Manifold::Dim,WeightingScheme::NumSigmaPoints>(0,0);
		augsigmapts=fullsigmapts.template block<AugManifold::Dim,WeightingScheme::NumSigmaPoints>(Manifold::Dim,0);
	}

	template<typename DestManifold, typename DestAugManifold=EuclideanManifold<0>, typename Functor>
	AugmentedUnscentedDistribution<DestManifold,DestAugManifold,WeightingScheme> transform(Functor f)
	{
		AugmentedUnscentedDistribution<DestManifold,DestAugManifold,WeightingScheme> ret;
		for (int i=0;i<WeightingScheme::NumSigmaPoints;i++)
			ret.sigmapts.col(i)=f(sigmapts.col(i),augsigmapts.col(i));
		if (DestAugManifold::Dim>0)
			ret.augsigmapts=augsigmapts.template block<DestAugManifold::Dim,WeightingScheme::NumSigmaPoints>(augsigmapts.rows()-DestAugManifold::Dim,0);
		ret.unscentedCombine();
		return ret;
	}
#if 0
	template<typename DestManifold, typename DestAugManifold, typename Functor>
	AugmentedUnscentedDistribution<DestManifold,DestAugManifold,WeightingScheme> transform(Functor f, Matrix<double,Manifold::DeltaDim,DestManifold::DeltaDim>& cross)
	{
		auto ret=transform<DestManifold,DestAugManifold,Functor>(f);

		cross.setZero();
		for (int i=0; i<WeightingScheme::NumSigmaPoints; i++)
			cross+=uscheme.unscentedCovWeight(i)*Manifold::manifoldSub(sigmapts.col(i),sqrtG.mu)*DestManifold::manifoldSub(ret.sigmapts.col(i),ret.sqrtG.mu).transpose();

		return ret;
	}
#endif
	const SqrtGaussian<Manifold>& toSqrtGaussian()
	{
		return sqrtG;
	}
//private:
	void unscentedCombine()
	{
		uscheme.template unscentedSqrtCombineManifold<Manifold>(sigmapts,sqrtG);
#if 0
		std::cout << "METHOD 1: \n" << sqrtG.covL << std::endl;
		//return;
		Matrix<double,Manifold::Dim+AugManifold::Dim,WeightingScheme::NumSigmaPoints> fullsigmapts;
		fullsigmapts.template block<Manifold::Dim,WeightingScheme::NumSigmaPoints>(0,0)=sigmapts;
		if (AugManifold::Dim)
			fullsigmapts.template block<AugManifold::Dim,WeightingScheme::NumSigmaPoints>(Manifold::Dim,0)=augsigmapts;
		SqrtGaussian<FullManifold> sg;
		uscheme.template unscentedSqrtCombineManifold<FullManifold>(fullsigmapts,sg);
		//sqrtG
		auto gg=sg.toGaussian();
		sqrtG.covL=gg.cov.template block<Manifold::Dim,Manifold::Dim>(0,0).llt().matrixL();
		sqrtG.mu=gg.mu.template segment<Manifold::Dim>(0);

		std::cout << "METHOD 2: \n" << sqrtG.covL << std::endl;
#endif
	}

	WeightingScheme uscheme;
	SqrtGaussian<Manifold> sqrtG;
	Matrix<double,Manifold::Dim,WeightingScheme::NumSigmaPoints> sigmapts;
	Matrix<double,AugManifold::Dim,WeightingScheme::NumSigmaPoints> augsigmapts;
};

template<typename Manifold_, typename AugManifold_, typename WeightingScheme_=Unscented::StandardScheme<ProductManifold<Manifold_,AugManifold_> > >
AugmentedUnscentedDistribution<Manifold_,AugManifold_,WeightingScheme_> make_augmentedUnscentedDistribution(const SqrtGaussian<Manifold_>& g,const SqrtGaussian<AugManifold_>& gaug) { return {g,gaug}; }

template<typename UD1, typename UD2>
Matrix<double,UD1::Manifold::DeltaDim,UD2::Manifold::DeltaDim> crossCov(const UD1& ud1, const UD2& ud2)
{
	Matrix<double,UD1::Manifold::DeltaDim,UD2::Manifold::DeltaDim> ret;
	ret.setZero();
	for (int i=0; i<UD1::WeightingScheme::NumSigmaPoints; i++)
		ret+=ud1.uscheme.unscentedCovWeight(i)*UD1::Manifold::manifoldSub(ud1.sigmapts.col(i),ud1.sqrtG.mu)*UD2::Manifold::manifoldSub(ud2.sigmapts.col(i),ud2.sqrtG.mu).transpose();
	return ret;
}

GSE_NS_END

#endif
