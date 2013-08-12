#ifndef GSE_RECBAYES_PARTICLEFILTER_H
#define GSE_RECBAYES_PARTICLEFILTER_H

#include "../GSEDefs.h"
#include <iostream>
#include <cassert>
#include <math.h>
#include "ParticleDistribution.h"
#include "../Distributions.h"

GSE_NS_BEGIN

//! @class ParticleFilter
//! @author Stephan Wirth, Frank kNeuhaus

template <typename PartDist>
class ParticleFilter
{
	PartDist& partDist;
public:

	ParticleFilter(PartDist& partDist_) : partDist(partDist_) {}

	template<typename SysModel, typename UserData>
	typename SysModel::StateVec sysModelAdvanceState(SysModel& sysModel, const typename SysModel::StateVec& state, const typename SysModel::SystemNoiseVec& noise, double dt, UserData& userdata)
	{
		return sysModel.advanceState(state,noise,dt,userdata);
	}
	template<typename SysModel>
	typename SysModel::StateVec sysModelAdvanceState(SysModel& sysModel, const typename SysModel::StateVec& state, const typename SysModel::SystemNoiseVec& noise, double dt, NoUserData&)
	{
		return sysModel.advanceState(state,noise,dt);
	}

	template<typename SysModel>
	void predict(SysModel& sysModel, const double dt)
	{
		partDist.checkResample();

		auto samp=make_distSampler(make_euclideanGaussian(sysModel.getNoise(dt)));

		// todo: sample here
		for (auto p : partDist.getParticles())
			p->state=sysModelAdvanceState(sysModel,p->state,samp.sample(),dt,p->userdata);
	}


	template<typename Observer, typename UserData>
	double measureObsModel(Observer& obs, const typename Observer::ObsVec& pred, UserData& userdata)
	{
		return obs.measure(pred,userdata);
	}
	template<typename Observer>
	double measureObsModel(Observer& obs, const typename Observer::ObsVec& pred, NoUserData&)
	{
		return obs.measure(pred);
	}

	template<typename ObsModel, typename UserData>
	typename ObsModel::ObsVec obsModelPredict(ObsModel& obsModel, const typename ObsModel::StateVec& state, const typename ObsModel::ObsNoiseVec& noise, UserData& userdata)
	{
		return obsModel.augmentedPredictMeasurement(state,noise,userdata);
	}
	template<typename ObsModel>
	typename ObsModel::ObsVec obsModelPredict(ObsModel& obsModel, const typename ObsModel::StateVec& state, const typename ObsModel::ObsNoiseVec& noise, NoUserData&)
	{
		return obsModel.augmentedPredictMeasurement(state,noise);
	}



	/**
	 * This method assigns weights to the particles using the observation model of the particle filter.
	 */
	template<typename SysModel, typename ObsModel, typename Observer>
	void predict_update(SysModel& sysModel, const double dt, ObsModel& obsModel, Observer &obs)
	{
		predict(sysModel,dt);

		auto samp=make_distSampler(make_euclideanGaussian(obs.getNoise()));

		for (auto p : partDist.getParticles())
		{
#if 0
			double weightSum=0;
			//auto pred=obsModel.augmentedPredictMeasurement(p->state,Observer::ObsNoiseVec::Zero());
			//double first=obs.measure(pred);
			//weightSum+=first;
			p->weight=0;

			double lowest=999999999;
			typename Observer::ObsNoiseVec lowsample;
			bool seta=false;

			for (int i=0;i<50;i++)
			{
				typename Observer::ObsNoiseVec sample;
				emv.nextSample(sample);

				auto pred=obsModel.augmentedPredictMeasurement(p->state,sample);
				double w=obs.measure(pred);
				/*if (w<lowest)
				{
					lowest=w;
					lowsample=sample;
					seta=1;
				}*/

				double c2=sample.transpose()*obs.getNoise().inverse()*sample;
				p->weight+=exp(-0.5*w);
			}
			//assert(seta);
			//double c2=lowsample.transpose()*obs.getNoise().inverse()*lowsample;
			//p->weight=1.0/sqrt((2*M_PI*obs.getNoise()).determinant())*exp(-0.5*c2)*exp(-0.5*lowest);
#else
			//p->weight=obs.measure(obsModel.augmentedPredictMeasurement(p->state,samp.sample()));
			p->weight=measureObsModel(obs,obsModelPredict(obsModel,p->state,samp.sample(),p->userdata),p->userdata);
#endif
		}

		partDist.update();
	}
};

template<typename Dist>
ParticleFilter<Dist> make_pf(Dist& pd) { return {pd}; }

GSE_NS_END

#endif

