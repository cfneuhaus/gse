#ifndef GSE_LS_GENERICABSCONSTRAINT_H
#define GSE_LS_GENERICABSCONSTRAINT_H

#include "../GSEDefs.h"
#include "NumDiff.h"

GSE_NS_BEGIN

template<typename Manifold_>
class GenericAbsConstraint
{
public:
	typedef typename ManifoldTraits<Manifold_>::Manifold Manifold;

	typedef boost::fusion::vector<RandomVariable<Manifold_> > ParamRVs;

	typename Manifold::DeltaVec error(const typename Manifold::Vec& xi,
										  typename Manifold::DeltaMat* dxi) const
	{
		typename Manifold::DeltaVec err=measInfLT*Manifold::manifoldSub(xi,meas);

		if (dxi)
		{
			auto jac=computeJacobian(*this,boost::fusion::make_vector(xi));
			(*dxi)=boost::fusion::at_c<0>(jac);
		}
		return err;
	}

	GenericAbsConstraint(ParamRVs params_, const typename Manifold::Vec& meas_, const typename Manifold::DeltaMat& measInf)
		: params(params_), meas(meas_), measInfLT(measInf.llt().matrixL().transpose())
	{
	}
	const ParamRVs& getParams() const { return params; }
private:
	ParamRVs params;
	typename Manifold::Vec meas;
	typename Manifold::DeltaMat measInfLT;
};

GSE_NS_END

#endif
