#ifndef GSE_POSEMANIFOLD3D_H
#define GSE_POSEMANIFOLD3D_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ManifoldUtil.h"

GSE_NS_BEGIN

template<typename Scalar_>
struct PoseManifold3D : public ProductManifold<EuclideanManifold<3,Scalar_>, SO3QuatManifold<Scalar_> >
{
	typedef Scalar_ Scalar;
	typedef ProductManifold<EuclideanManifold<3,Scalar_>, SO3QuatManifold<Scalar_> > Base;
	template<typename T>
	struct Other
	{
		typedef PoseManifold3D<T> Type;
	};

private:
	typedef Eigen::Quaternion<Scalar> TQuat;
	typedef Matrix<Scalar,3,1> TVec3;
public:

	static typename Base::Vec zeroPose() { typename Base::Vec ret; ret<< 0,0,0,1,0,0,0; return ret; }

#define Tsegment template segment
	static typename Base::Vec poseMinus(const typename Base::Vec& x, const typename Base::Vec& y)
	{
		TQuat qx(x(3),x(4),x(5),x(6));
		TQuat qy(y(3),y(4),y(5),y(6));
		TQuat qy_conj(y(3),-1.0*y(4),-1.0*y(5),-1.0*y(6));
		TQuat qres=qy_conj*qx;
		typename Base::Vec ret;
		ret.Tsegment<3>(0)=rotatePt(qy_conj,x.Tsegment<3>(0)-y.Tsegment<3>(0));
		ret(3)=qres.w();	ret(4)=qres.x();	ret(5)=qres.y();	ret(6)=qres.z();
		return ret;
	}
	// not for actual optimization
	static typename Base::Vec poseAdd(const typename Base::Vec& x, const typename Base::Vec& d)
	{
		TQuat qx(x(3),x(4),x(5),x(6));
		TQuat qd(d(3),d(4),d(5),d(6));
		TQuat qres=qx*qd;
		typename Base::Vec ret;
		ret.Tsegment<3>(0)=x.Tsegment<3>(0)+rotatePt(qx,d.Tsegment<3>(0));
		ret(3)=qres.w();	ret(4)=qres.x();	ret(5)=qres.y();	ret(6)=qres.z();
		return ret;
	}
private:

	// for poseMinus...
	static TVec3 rotatePt(const TQuat& q, const TVec3& v)
	{
		Scalar x=q.x();
		Scalar y=q.y();
		Scalar z=q.z();
		Scalar w=q.w();

		Scalar
			//ww=w*w,
			wx=w*x, xx=x*x,
			wy=w*y, xy=x*y, yy=y*y,
			wz=w*z, xz=x*z, yz=y*z, zz=z*z;

		Scalar X= v(0), Y=v(1), Z=v(2);
		TVec3 ret;
		ret(0) = (1.0+(-2.0*zz)+(-2.0*yy))*X + 2*(xy-wz)*Y + 2*(wy+xz)*Z;
		ret(1) = 2*(xy+wz)*X + (1.0+(-2*zz)+(-2*xx))*Y + 2*(yz-wx)*Z;
		ret(2) = 2*(xz-wy)*X + 2*(yz+wx)*Y + (1.0+(-2.0*yy)+(-2.0*xx))*Z;
		return ret;
	}
#undef Tsegment
};

typedef PoseManifold3D<double> PoseManifold3Dd;

namespace
{
void poseToRt(const PoseManifold3Dd::Vec& pose, Matrix3d& R, Vector3d& t)
{
	Quaterniond quat(pose(3),pose(4),pose(5),pose(6));

	Transform<double,3,Affine> trans;
	trans=quat;

	t=pose.segment<3>(0);
	R=trans.matrix().block<3,3>(0,0);
}

PoseManifold3Dd::Vec RtToPose(const Matrix3d& R, const Vector3d& t)
{
	Quaterniond q(R);
	PoseManifold3Dd::Vec p;
	p.segment<3>(0)=t;
	p(3)=q.w(); p(4)=q.x(); p(5)=q.y(); p(6)=q.z();
	return p;
}

void poseToQt(const PoseManifold3Dd::Vec& pose, Quaterniond& q, Vector3d& t)
{
	q=Quaterniond(pose(3),pose(4),pose(5),pose(6));
	t=pose.segment<3>(0);
}

PoseManifold3Dd::Vec QtToPose(const Quaterniond& q, const Vector3d& t)
{
	PoseManifold3Dd::Vec p;
	p.segment<3>(0)=t;
	p(3)=q.w(); p(4)=q.x(); p(5)=q.y(); p(6)=q.z();
	return p;
}

Matrix4d poseToTransform(const PoseManifold3Dd::Vec& p)
{
	Matrix3d R;
	Vector3d t;
	poseToRt(p,R,t);
	Matrix4d ret=Matrix4d::Identity();
	ret.block<3,3>(0,0)=R;
	ret.block<3,1>(0,3)=t;
	return ret;
}
PoseManifold3Dd::Vec affineToPose(const Eigen::Affine3d& trans)
{
	Quaterniond q(trans.matrix().block<3,3>(0,0));
	PoseManifold3Dd::Vec p;
	p.segment<3>(0)=trans.matrix().block<3,1>(0,3);
	p(3)=q.w(); p(4)=q.x(); p(5)=q.y(); p(6)=q.z();
	return p;
}
Eigen::Affine3d poseToAffine(const PoseManifold3Dd::Vec& pose)
{
	Eigen::Affine3d trans = Eigen::Affine3d::Identity();
	// hack, since eigen doesn't like poseToRt(trans.matrix().block<...>()...)
	Eigen::Matrix3d R;
	Eigen::Vector3d t;
	poseToRt(pose,R, t);
	trans.matrix().block<3,3>(0,0)=R;
	trans.matrix().block<3,1>(0,3)=t;
	return trans;
}

}

GSE_NS_END

#endif
