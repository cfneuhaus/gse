#ifndef GSE_RECBAYES_SQRT_UKF_MANIFOLD_H
#define GSE_RECBAYES_SQRT_UKF_MANIFOLD_H

#include "../GSEDefs.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "Unscented.h"
#include "../ManifoldUtil.h"

GSE_NS_BEGIN

using namespace Eigen;

template<typename ManifoldType_, int SystemNoiseDim_>
struct SysModelBase
{
	typedef ManifoldType_ ManifoldType;
	static const unsigned StateDim = ManifoldType::Dim;
	static const unsigned StateDeltaDim = ManifoldType::DeltaDim;
	static const unsigned SystemNoiseDim = SystemNoiseDim_;

	typedef Matrix<double, StateDim, 1> StateVec;
	typedef Matrix<double, SystemNoiseDim, 1> SystemNoiseVec;
	typedef Matrix<double, SystemNoiseDim, SystemNoiseDim> SystemNoiseMat;
};

template<typename StateManifoldType_, typename ManifoldType_, int ObsNoiseDim_>
struct ObsModelBase
{
	typedef ManifoldType_ ManifoldType;

	static const unsigned StateDim = StateManifoldType_::Dim;
	typedef Matrix<double, StateDim, 1> StateVec;

	static const unsigned ObsDim = ManifoldType::Dim;
	static const unsigned ObsDeltaDim = ManifoldType::DeltaDim;
	static const unsigned ObsNoiseDim = ObsNoiseDim_;

	typedef Matrix<double, ObsDim, 1> ObsVec;
	typedef Matrix<double, ObsNoiseDim, 1> ObsNoiseVec;
	typedef Matrix<double, ObsNoiseDim, ObsNoiseDim> ObsNoiseMat;
};

template<typename ManifoldType_, int ObsNoiseDim_>
struct ObsBase
{
	typedef ManifoldType_ ManifoldType;

	static const unsigned ObsDim = ManifoldType::Dim;
	static const unsigned ObsDeltaDim = ManifoldType::DeltaDim;
	static const unsigned ObsNoiseDim = ObsNoiseDim_;

	typedef Matrix<double, ObsDim, 1> ObsVec;
	typedef Matrix<double, ObsDeltaDim, ObsDeltaDim> ObsMat;
	typedef Matrix<double, ObsNoiseDim, 1> ObsNoiseVec;
	typedef Matrix<double, ObsNoiseDim, ObsNoiseDim> ObsNoiseMat;
};


template<typename StateManifold>
class SqrtUKFManifold
{
public:
	typedef Matrix<double,StateManifold::Dim,1> StateVec;
	typedef Matrix<double,StateManifold::DeltaDim,StateManifold::DeltaDim> StateMat;

	SqrtUKFManifold(const StateVec& mu_, const StateMat& sqrtCov_);

	template<typename SysModel, typename ObsModel, typename Observer>
	void predict_update(SysModel& sysModel, const double dt, ObsModel& obsModel, Observer& obs);

	template<typename SysModel>
	void predict(SysModel& sysModel, const double dt);

	const StateVec& getState() const { return mu; }
	const StateMat& getSqrtCov() const { return sqrtCov; }
	StateMat computeCov() const { return sqrtCov*sqrtCov.transpose(); }

	void setState(const StateVec& mu_, const StateMat& sqrtCov_)
	{
		mu=mu_;
		sqrtCov=sqrtCov_;
	}

private:
	StateVec mu;
	// Lower Diagonal of Cov... Cov=sqrtCov*sqrtCov'
	StateMat sqrtCov;
};

//-----------------------------------------------------------------------------
template<typename StateManifold>
SqrtUKFManifold<StateManifold>::SqrtUKFManifold(const StateVec& mu_, const StateMat& sqrtCov_) : mu(mu_), sqrtCov(sqrtCov_)
{
}
//-----------------------------------------------------------------------------
template<typename StateManifold_> template<typename SysModel>
void SqrtUKFManifold<StateManifold_>::predict(SysModel& sysModel, const double dt)
{
	const int StateDim=SysModel::StateDim;
	const int StateDeltaDim=SysModel::StateDeltaDim;
	const int SystemNoiseDim=SysModel::SystemNoiseDim;

	const int AugStateDim=StateDim+SystemNoiseDim;
	const int AugStateDeltaDim=StateDeltaDim+SystemNoiseDim;

	typedef Matrix<double,StateDim,1> StateVec;
	typedef Matrix<double,StateDeltaDim,StateDeltaDim> StateMat;
	typedef Matrix<double,SystemNoiseDim,1> SystemNoiseVec;

	typedef Matrix<double,AugStateDim,1> AugStateVec;
	typedef Matrix<double,AugStateDeltaDim,AugStateDeltaDim> AugStateMat;

	typedef typename SysModel::ManifoldType StateManifold;

	AugStateVec xak=AugStateVec::Zero();
	xak.segment(0,StateDim)=mu;

	AugStateMat sqrtPak=AugStateMat::Zero();
	sqrtPak.template block<StateDeltaDim,StateDeltaDim>(0,0)=sqrtCov;
	sqrtPak.template block<SystemNoiseDim,SystemNoiseDim>(StateDeltaDim,StateDeltaDim)=sysModel.getNoise(dt).llt().matrixL(); // a bit hacky

	typedef ProductManifold<StateManifold,EuclideanManifold<SystemNoiseDim> > AugmentedStateManifold;
	typedef Unscented::StandardScheme<AugmentedStateManifold> UScheme;
	const int NumSigmaPoints=UScheme::NumSigmaPoints;
	UScheme uscheme;

	Matrix<double,AugStateDim,NumSigmaPoints> chis=uscheme.unscentedSqrtTransform(xak,sqrtPak);
	for (int i=0; i<NumSigmaPoints; i++)
	{
		StateVec state=chis.col(i).segment(0,StateDim);
		SystemNoiseVec systemNoise=chis.col(i).segment(StateDim,SystemNoiseDim);
		chis.col(i).segment(0,StateDim)=sysModel.advanceState(state,systemNoise,dt);
	}

	Matrix<double,StateDim,NumSigmaPoints> tmpChis=chis.template block<StateDim,NumSigmaPoints>(0,0);

	uscheme.template unscentedSqrtCombineManifold<StateManifold>(tmpChis,mu,sqrtCov);

}
//-----------------------------------------------------------------------------
template<typename StateManifold_> template<typename SysModel, typename ObsModel, typename Observer>
void SqrtUKFManifold<StateManifold_>::predict_update(SysModel& sysModel, const double dt, ObsModel& obsModel, Observer &obs)
{
	const int StateDim=SysModel::StateDim;
	const int StateDeltaDim=SysModel::StateDeltaDim;
	const int SystemNoiseDim=SysModel::SystemNoiseDim;

	const int ObsDim=ObsModel::ObsDim;
	const int ObsDeltaDim=ObsModel::ObsDeltaDim;
	const int ObsNoiseDim=ObsModel::ObsNoiseDim;

	const int AugStateDim=StateDim+SystemNoiseDim+ObsNoiseDim;
	const int AugStateDeltaDim=StateDeltaDim+SystemNoiseDim+ObsNoiseDim;

	typedef Matrix<double,StateDim,1> StateVec;
	typedef Matrix<double,StateDeltaDim,StateDeltaDim> StateMat;
	typedef Matrix<double,SystemNoiseDim,1> SystemNoiseVec;

	typedef Matrix<double,ObsDim,1> ObsVec;
	typedef Matrix<double,ObsDeltaDim,ObsDeltaDim> ObsMat;
	typedef Matrix<double,ObsNoiseDim,1> ObsNoiseVec;

	typedef Matrix<double,AugStateDim,1> AugStateVec;
	typedef Matrix<double,AugStateDeltaDim,AugStateDeltaDim> AugStateMat;

	typedef typename SysModel::ManifoldType StateManifold;
	typedef typename ObsModel::ManifoldType ObsManifold;

	AugStateVec xak=AugStateVec::Zero();
	xak.segment(0,StateDim)=mu;

	AugStateMat sqrtPak=AugStateMat::Zero();
	sqrtPak.template block<StateDeltaDim,StateDeltaDim>(0,0)=sqrtCov;
	sqrtPak.template block<SystemNoiseDim,SystemNoiseDim>(StateDeltaDim,StateDeltaDim)=sysModel.getNoise(dt).llt().matrixL(); // a bit hacky
	sqrtPak.template block<ObsNoiseDim,ObsNoiseDim>(StateDeltaDim+SystemNoiseDim,StateDeltaDim+SystemNoiseDim)=obs.getNoise().llt().matrixL(); // a bit hacky

	typedef ProductManifold<StateManifold,EuclideanManifold<SystemNoiseDim+ObsNoiseDim> > AugmentedStateManifold;
	typedef Unscented::StandardScheme<AugmentedStateManifold> UScheme;
	const int NumSigmaPoints=UScheme::NumSigmaPoints;
	UScheme uscheme;

	Matrix<double,AugStateDim,NumSigmaPoints> chis=uscheme.unscentedSqrtTransform(xak,sqrtPak);
	for (int i=0; i<NumSigmaPoints; i++)
	{
		StateVec state=chis.col(i).segment(0,StateDim);
		SystemNoiseVec systemNoise=chis.col(i).segment(StateDim,SystemNoiseDim);
		chis.col(i).segment(0,StateDim)=sysModel.advanceState(state,systemNoise,dt);
	}

	Matrix<double,StateDim,NumSigmaPoints> tmpChis=chis.template block<StateDim,NumSigmaPoints>(0,0);

	uscheme.template unscentedSqrtCombineManifold<StateManifold>(tmpChis,mu,sqrtCov);

	//////////////////////////////////////// UPDATE

	Matrix<double,ObsDim,NumSigmaPoints> gammas;

	for (int i=0; i<NumSigmaPoints; i++)
	{
		StateVec state=chis.col(i).segment(0,StateDim);
		ObsNoiseVec obsNoise=chis.col(i).segment(StateDim+SystemNoiseDim,ObsNoiseDim);
		gammas.col(i)=obsModel.augmentedPredictMeasurement(state,obsNoise);
	}

	ObsMat sqrtS;
	ObsVec z;
	uscheme.template unscentedSqrtCombineManifold<ObsManifold>(gammas,z,sqrtS);

	ObsVec m;
	if (!obs.measure(z,sqrtS*sqrtS.transpose(),m)) // a bit hacky
	{
		std::cout << "Measurement failed! Skipping Kalman Update" << std::endl;
		return;
	}


	Matrix<double,StateDeltaDim,ObsDeltaDim> cross=Matrix<double,StateDeltaDim,ObsDeltaDim>::Zero();
	for (int i=0; i<NumSigmaPoints; i++)
		cross+=uscheme.unscentedCovWeight(i)*StateManifold::manifoldSub(tmpChis.col(i),mu)*ObsManifold::manifoldSub(gammas.col(i),z).transpose();

	//ObsVec small=ObsVec::Ones(ObsDim)*0.000001;
	//sqrtS+=small.asDiagonal();

	Matrix<double,ObsDeltaDim,StateDeltaDim> crossT=cross.transpose();
	sqrtS.template triangularView<Eigen::Lower>().solveInPlace(crossT);
	sqrtS.transpose().template triangularView<Eigen::Upper>().solveInPlace(crossT);

	Matrix<double,StateDeltaDim,ObsDeltaDim> K=crossT.transpose();

	mu=StateManifold::manifoldAdd(mu,K*ObsManifold::manifoldSub(m,z));

	Matrix<double,StateDeltaDim,ObsDeltaDim> U=K*sqrtS;

	//	// not sure if this really requires a loop...
	for (int i=0;i<U.cols();i++)
	{
		if (internal::llt_inplace<double,Lower>::rankUpdate(sqrtCov,U.col(i),-1)>=0)
		{
			//std::cout << "Numerical Issue in RankUpdate2" << std::endl;
		}
	}

}

GSE_NS_END

#endif
