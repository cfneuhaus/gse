#ifndef GSE_POSEMANIFOLD2D_H
#define GSE_POSEMANIFOLD2D_H

#include"GSEDefs.h"
#include <Eigen/Core>
#include "ManifoldUtil.h"

GSE_NS_BEGIN

template<typename Scalar_>
struct PoseManifold2D : public ProductManifold<EuclideanManifold<2,Scalar_>, SO2Manifold<Scalar_> >
{
	typedef ProductManifold<EuclideanManifold<2,Scalar_>, SO2Manifold<Scalar_> > Base;
	template<typename T>
	struct Other
	{
		typedef PoseManifold2D<T> Type;
	};

	static typename Base::Vec zeroPose() { return Base::Vec::Zero(); }

#define Tsegment template segment
	static typename Base::Vec poseMinus(const typename Base::Vec& x, const typename Base::Vec& y)
	{
		typename Base::Vec ret;
		ret(0)=     (x(0)-y(0))*cos(y(2))+(x(1)-y(1))*sin(y(2));
		ret(1)=-1.0*(x(0)-y(0))*sin(y(2))+(x(1)-y(1))*cos(y(2));
		ret(2)=x(2)-y(2);
		return ret;
	}
	// not for actual optimization
	static typename Base::Vec poseAdd(const typename Base::Vec& x, const typename Base::Vec& d)
	{
		typename Base::Vec ret;
		ret(0)=x(0)+d(0)*cos(x(2))-d(1)*sin(x(2));
		ret(1)=x(1)+d(0)*sin(x(2))+d(1)*cos(x(2));
		ret(2)=x(2)+d(2);
		return ret;
	}
#undef Tsegment
};

typedef PoseManifold2D<double> PoseManifold2Dd;

namespace
{
void poseToRt(const PoseManifold2Dd::Vec& pose, Matrix2d& R, Vector2d& t)
{
	const double c=cos(pose(2));
	const double s=sin(pose(2));
	R << c, -s,
			s,  c;
	t(0)=pose(0);
	t(1)=pose(1);
}

PoseManifold2Dd::Vec RtToPose(const Matrix2d& R, const Vector2d& t)
{
	PoseManifold2Dd::Vec ret;
	ret(0)=t(0);
	ret(1)=t(1);
	ret(2)=atan2(R(1,0),R(0,0));
	return ret;
}
}

GSE_NS_END

#endif // POSEMANIFOLD2D_H
