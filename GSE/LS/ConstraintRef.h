#ifndef GSE_LS_CONSTRAINTREF_H
#define GSE_LS_CONSTRAINTREF_H

#include "../GSEDefs.h"
#include "../ManifoldUtil.h"
#include <boost/fusion/mpl.hpp>

GSE_NS_BEGIN

template<typename ConstraintType_>
class ConstraintRef
{
public:
	typedef ConstraintType_ ConstraintType;
	ConstraintRef() : id(-1) {}
	ConstraintRef(int id_) : id(id_) {}
	int getId() const { return id; }
	bool operator<(const ConstraintRef<ConstraintType_>& other) const
	{
		return id<other.id;
	}
	bool operator==(const ConstraintRef<ConstraintType_>& other) const
	{
		return id==other.id;
	}
private:
	int id;
};

GSE_NS_END

// allow ConstraintRef to be used in unordered_maps
namespace std
{
template <typename T>
class hash<GSE::ConstraintRef<T> >{
public :
	size_t operator()(const GSE::ConstraintRef<T>& x) const
	{
		return hash<int>()(x.getId());
	}
};
}

#endif
