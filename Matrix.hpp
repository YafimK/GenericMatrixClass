/**
 * @file	Matrix.hpp
 * @author  fimak
 * @brief	Declares and implements the genric matrix class.
 */

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include <numeric>
#include <exception>
#include "Complex.h"

#ifndef _MATRIX_HPP
#define _MATRIX_HPP

/** @brief	The row copy start position. */
static const int ROW_COPY_START_POS = 0;
/** @brief	Message og the exception in case of the diffrent matrice size. */
static const std::string DIFFRENT_MATRICE_SIZE_MSG = "cannot addition matrices of different sizes.";
/** @brief	The default matrix row size. */
const static unsigned int DEFAULT_MATRIX_ROW_SIZE = 1;
/** @brief	The default matrix column size. */
const static unsigned int DEFAULT_MATRIX_COLUMN_SIZE = 1;
/** @brief	The default trace result. */
const static unsigned int DEFAULT_TRACE_RESULT = 0;
/** @brief	The default test result. */
const static bool DEFAULT_TEST_RESULT = false;


/** @brief	The print matrix delimiter. */
static const std::string PRINT_MATRIX_DELIM = "\t";
/** @brief	The default empty string. */
static const std::string DEFAULT_EMPTY_STRING = "";
/** @brief	Message describing the matrix is not square as required. */
static const std::string MATRIX_NOT_SQUARE_MSG = "Matrix is not square as required";
/** @brief	The default matrix cell value. */
static const int DEFAULT_MATRIX_CELL_VAL = 0;



/**
 * @class	Matrix
 *
 * @brief	A generic matrix implementation
 *
 * @tparam	T	Generic type parameter.
 */

template <class T>
class Matrix
{
	/**
	* @def	PARALLAL_STATUS_LABEL();
	*
	* @brief	A macro that defines parallal status label.
	*/

	#define PARALLAL_STATUS_LABEL "parallel"

	/**
	* @def	NON_PARALLEL_STATUS_LABEL();
	*
	* @brief	A macro that defines non parallel status label.
	*/

	#define NON_PARALLEL_STATUS_LABEL "non-parallel"

	/**
	* @def	DEFAULT_PARALLEL_MODE_CHANGE_MSG();
	*
	* @brief	A macro that defines default parallel mode change message.
	*/

	#define DEFAULT_PARALLEL_MODE_CHANGE_MSG "Generic matrix mode changed to " << parallelStatus \
	<< " mode."

	/**
	* @def	ILLEGAL_MATRIX_SIZE_MSG();
	*
	* @brief	A macro that defines illegal matrix size message.
	*
	* @param	rows	The rows.
	*/

	#define ILLEGAL_MATRIX_SIZE_MSG "Matrix size is illegal! you cannot declere a matrix with " + \
	std::to_string(rows) + " rows and " + std::to_string(cols) + " columns"

	/**
	* @def	REQUESTED_INDICES_OUT_BOUND_MSG();
	*
	* @brief	A macro that defines requested indices out bound message.
	*
	* @param	row	The row.
	*/

	#define REQUESTED_INDICES_OUT_BOUND_MSG "The requested matrix indices [" + std::to_string \
	(row) +	"][" + std::to_string(column) + "] are not in range"

	/**
	* @def	MULTIPLIED_MATRICE_WITH_WRONG_SIZES_MSG();
	*
	* @brief	A macro that defines multiplied matrice with wrong sizes message.
	*
	* @param	this->cols()	the colums number of the left matrix
	* @param	rightMatrix.rows()	the rows number of the right matrix
	*/

	#define MULTIPLIED_MATRICE_WITH_WRONG_SIZES_MSG "This matrice cannot be multiplies since you " \
"cannot multiplty a matrix with " + std::to_string(this->cols()) + " colums by matrix " \
"with " + std::to_string(rightMatrix.rows()) + " rows"


private:
	/** @brief	true if this instance is parallel. */
	static bool s_isParallel;
	/** @brief	The matrix. */
	std::vector<T> _matrix;
	/** @brief	Size of the row. */
	unsigned int _rowSize;
	/** @brief	Size of the column. */
	unsigned int _columnSize;

	/**
	 * @fn	void Matrix::_initilizeMatrix();
	 *
	 * @brief	Initialize matrix helper method.
	 */

	void _initilizeMatrix();

	/**
	 * @fn	void Matrix::_initilizeMatrix(unsigned int rows, unsigned int cols);
	 *
	 * @brief	Initilize matrix with specified cols and rows
	 *
	 * @param	rows	The rows.
	 * @param	cols	The cols.
	 */

	void _initilizeMatrix(unsigned int rows, unsigned int cols);

	/**
	 * @fn	void Matrix::_regularAdditionAlgo(const Matrix<T>& source, Matrix<T>& target) const;
	 *
	 * @brief	Regular addition algo.
	 *
	 * @param	source		  	the left matrix for the addition
	 * @param [in,out]	target	the result matrix
	 */

	void _regularAdditionAlgo(const Matrix<T>& source, Matrix<T>& target) const;

	/**
	 * @fn	void Matrix::_multiThreadedAdditionAlgo(const Matrix<T>& source, Matrix<T>& target) const;
	 *
	 * @brief	Multi threaded addition algo.
	 *
	 * @param	source		  	the left matrix for the addition
	 * @param [in,out]	target	the result matrix.
	 */

	void _multiThreadedAdditionAlgo(const Matrix<T>& source, Matrix<T>& target) const;

	/**
	 * @fn	void Matrix::_regularMultAlgo(const Matrix<T>& source, Matrix<T>& target) const;
	 *
	 * @brief	Regular multiply algo.
	 *
	 * @param	source		  	the left matrix for the addition
	 * @param [in,out]	target	the result matrix.
	 */

	void _regularMultAlgo(const Matrix<T>& source, Matrix<T>& target) const;

	/**
	 * @fn	void Matrix::_multiThreadedMultAlgo(const Matrix<T>& source, Matrix<T>& target) const;
	 *
	 * @brief	Multi threaded multiply algo.
	 *
	 * @param	source		  	the left matrix for the addition
	 * @param [in,out]	target	the result matrix.
	 */

	void _multiThreadedMultAlgo(const Matrix<T>& source, Matrix<T>& target) const;

	/**
	 * @fn	void Matrix::_calculatestd::vectorTostd::vectorAdd (const Matrix<T>& lhs, const Matrix<T>& rhs, Matrix<T>& result, const int& rowNum) const;
	 *
	 * @brief	Calculates the result of an addition OP between two vectros
	 *
	 * @param	lhs			  	The left hand matrix.
	 * @param	rhs			  	The right hand matrix.
	 * @param [in,out]	result	The result.
	 * @param	rowNum		  	The row number.
	 */

	void _calculateVectorToVectorAdd
			(const Matrix<T>& lhs, const Matrix<T>& rhs, Matrix<T>& result, const int& rowNum)
			const;

	/**
	 * @fn	void Matrix::_isMatrixSizeLegit(const unsigned int& rows, const unsigned int& cols) const;
	 *
	 * @brief	checks if matrix size legal
	 *
	 * @param	rows	The rows.
	 * @param	cols	The cols.
	 */

	void _isMatrixSizeLegit(const unsigned int& rows, const unsigned int& cols) const;

	/**
	 * @fn	void Matrix::_isIndicesOutOfRange(const unsigned int& row, const unsigned int& column) const;
	 *
	 * @brief	checks if indices out of range.
	 *
	 * @param	row   	The row.
	 * @param	column	The column.
	 */

	void _isIndicesOutOfRange(const unsigned int& row, const unsigned int& column) const;

	/**
	 * @fn	void Matrix::_isMatrixSizeEqual(const Matrix& rightMatrix) const;
	 *
	 * @brief	Is matrix size equal.
	 *
	 * @param	rightMatrix	The right matrix.
	 */

	void _isMatrixSizeEqual(const Matrix& rightMatrix) const;

	/**
	 * @fn	void Matrix::_isMultipliedMatriceSizeLegit(const Matrix& rightMatrix) const;
	 *
	 * @brief	Is multiplied matrice left matrix cols = right matrix rows.
	 *
	 * @param	rightMatrix	The right matrix.
	 */

	void _isMultipliedMatriceSizeLegit(const Matrix& rightMatrix) const;

	/**
	 * @fn	void Matrix::_isVectorFitsMatrix(const size_t& std::vectorSize) const;
	 *
	 * @brief	Is std::vector fits matrix.
	 *
	 * @param	std::vectorSize	Size of the std::vector.
	 */

	void _isVectorFitsMatrix(const unsigned long& vectorSize) const;

	/**
	 * @fn	void Matrix::_multiplyRowByCol (const std::vector<std::vector<T>>& columns, const
	 * std::vector<T>& row, std::vector<T>& resultMatrixRow, const unsigned int&
	 * numOfLeftMatrixColums) const;
	 *
	 * @brief	Multiplies row by col.
	 *
	 * @param	columns				   	The right matrix seperated columns
	 * @param	row					   	The current row to be multiplied on all the columns
	 * @param [in,out]	resultMatrixRow	The result matrix row.
	 * @param	numOfLeftMatrixColums  	Number of left matrix colums.
	 */

	void _multiplyRowByCol
			(const std::vector<std::vector<T>>& columns, const std::vector<T>& row, std::vector<T>&
			 resultMatrixRow, const unsigned int& numOfLeftMatrixColums) const;

	//**The next two op function had to be made due to some porblem with the complex class using
	// std::plus, std::multiply */

	/**
	 * @fn	static T Matrix::_productOp(const T& leftItem, const T& rightItem);
	 *
	 * @brief	Product operation.
	 *
	 * @param	leftItem 	The left item.
	 * @param	rightItem	The right item.
	 *
	 * @return	the product of two items.
	 */

	static T _productOp(const T& leftItem, const T& rightItem);

	/**
	 * @fn	static T Matrix::_sumOp(const T& leftItem, const T& rightItem);
	 *
	 * @brief	Sum operation.
	 *
	 * @param	leftItem 	The left item.
	 * @param	rightItem	The right item.
	 *
	 * @return	The total number both items.
	 */

	static T _sumOp(const T& leftItem, const T& rightItem);

	/**
	 * @fn	void Matrix::_rowToMatrix(const std::vector<T>& row, std::vector<T>& targetMatrix, const
	 * unsigned int rowNumber) const;
	 *
	 * @brief	insert a std::vector row to a matrix represented by a std::vector
	 *
	 * @param	row						The row.
	 * @param [in,out]	targetMatrix	Target matrix.
	 * @param	rowNumber				The row number.
	 */

	void _rowToMatrix(const std::vector<T>& row, std::vector<T>& targetMatrix, const unsigned int
					  rowNumber) const;

	/**
	 * @fn	void Matrix::_vectorColumnSlicer(const Matrix<T>& source1, std::vector<std::vector<T>>&
	 * target) const;
	 *
	 * @brief	turns a matrix std::vector to std::vector<std::vector<T>> reperesnting a std::vector
	 * of colums
	 *
	 * @param	source1		  	Source 1.
	 * @param [in,out]	target	Target for the.
	 */

	void _vectorColumnSlicer(const Matrix<T>& source1, std::vector<std::vector<T>>& target) const;

	/**
	 * @fn	static unsigned int Matrix::_calculateCellNumber (const unsigned int& rowNumber, const
	 * unsigned int& columnNumber, const unsigned int& cols);
	 *
	 * @brief	Calculates the rewuired cell number position in the std::vector
	 *
	 * @param	rowNumber   	The row number.
	 * @param	columnNumber	The column number.
	 * @param	cols			The cols.
	 *
	 * @return	The calculated cell number.
	 */

	static unsigned int _calculateCellNumber (const unsigned int& rowNumber, const unsigned int&
											  columnNumber, const unsigned int& cols);


public:

	/**
	 * @fn	Matrix::Matrix(const unsigned int rowSize, const unsigned int columSize);
	 *
	 * @brief	Constructor.
	 *
	 * @param	rowSize  	Size of the row.
	 * @param	columSize	Size of the colum.
	 */

	Matrix(const unsigned int rowSize, const unsigned int columSize);

	/**
	 * @fn	Matrix::Matrix();
	 *
	 * @brief	Default constructor.
	 */

	Matrix();

	/**
	 * @fn	Matrix::Matrix(const Matrix& source);
	 *
	 * @brief	Copy constructor.
	 *
	 * @param	source	Source for the.
	 */

	Matrix(const Matrix& source);

	/**
	 * @fn	Matrix::Matrix(Matrix && source);
	 *
	 * @brief	Move constructor.
	 *
	 * @param [in,out]	source	Source for the.
	 */

	Matrix(Matrix && source);

	/**
	 * @fn	Matrix::Matrix(unsigned int rowSize, unsigned int columnSize, const std::vector<T>&
	 *  cells);
	 *
	 * @brief	Constructor.
	 *
	 * @param	rowSize   	Size of the row.
	 * @param	columnSize	Size of the column.
	 * @param	cells	  	std::vector to fill the matrix
	 */

	Matrix(unsigned int rowSize, unsigned int columnSize, const std::vector<T>& cells);

	/**
	 * @fn	unsigned int Matrix::rows() const
	 *
	 * @brief	Gets the rows.
	 *
	 * @return number of rows
	 */

	unsigned int rows() const
	{
		return _rowSize;
	}

	/**
	 * @fn	std::vector<T> Matrix::getMatrix() const
	 *
	 * @brief	Gets the matrix.
	 *
	 * @return	The matrix.
	 */

	std::vector<T> getMatrix() const
	{
		return _matrix;
	}

	/**
	 * @fn	unsigned int Matrix::cols() const
	 *
	 * @brief	Gets the cols  number
	 *
	 * @return	number of columns
	 */

	unsigned int cols() const
	{
		return _columnSize;
	}

	/**
	 * @fn	friend bool Matrix::operator!=(const Matrix& rhs, const Matrix& lhs)
	 *
	 * @brief	Inequality operator.
	 *
	 * @param	rhs	The first instance to compare.
	 * @param	lhs	The second instance to compare.
	 *
	 * @return	true if the parameters are not considered equivalent.
	 */

	friend bool operator!=(const Matrix& rhs, const Matrix& lhs)
	{
		return (! (lhs == rhs));
	}

	/**
	 * @fn	friend bool Matrix::operator==(const Matrix& lhs, const Matrix& rhs)
	 *
	 * @brief	Equality operator.
	 *
	 * @param	lhs	The first instance to compare.
	 * @param	rhs	The second instance to compare.
	 *
	 * @return	true if the parameters are considered equivalent.
	 */

	friend bool operator==(const Matrix& lhs, const Matrix& rhs)
	{
		if (lhs.getMatrix() == rhs.getMatrix())
		{
			if (lhs.rows() == rhs.rows())
			{
				if (lhs.cols() == rhs.cols())
				{
					return true;
				}
			}
		}

		return DEFAULT_TEST_RESULT;
	}

	/**
	 * @fn	friend std::ostream& Matrix::operator<<(std::ostream& os, const Matrix<T>& source)
	 *
	 * @brief	Stream insertion operator.
	 *
	 * @param [in,out]	os	The operating system.
	 * @param	source	  	Source for the.
	 *
	 * @return	The shifted result.
	 */

	friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& source)
	{
		unsigned int colCounter = 0;
		for (T item: source.getMatrix())
		{
			os << item << PRINT_MATRIX_DELIM;
			colCounter ++;
			if (colCounter == source.cols())
			{
				os << std::endl;
				colCounter = 0;
			}
		}
		return os;
	}

	/**
	 * @fn	Matrix<T> Matrix::trans();
	 *
	 * @brief	Gets the transpose of the matrix
	 *
	 * @return	transposed matrix;
	 */

	Matrix<T> trans();

	/**
	 * @fn	const T Matrix::trace();
	 *
	 * @brief	Gets the matrix trace
	 *
	 * @return	matrix trace
	 */

	const T trace();

	/**
	 * @fn	Matrix::~Matrix() = default;
	 *
	 * @brief	Destructor.
	 */

	~Matrix() {};

	/**
	 * @fn	bool Matrix::isSquareMatrix() const;
	 *
	 * @brief	Query if this matrix is square matrix.
	 *
	 * @return	true if square matrix, false if not.
	 */

	bool isSquareMatrix() const;

	/**
	 * @fn	void Matrix::setMatrix(const std::vector<T>& matrix);
	 *
	 * @brief	Sets a matrix.
	 *
	 * @param	matrix	The matrix.
	 */

	void setMatrix(const std::vector<T>& matrix);

	/**
	 * @fn	Matrix<T> Matrix::operator*(const Matrix<T>& source) const;
	 *
	 * @brief	Multiplication operator.
	 *
	 * @param	source	Source for the.
	 *
	 * @return	The result of the operation.
	 */

	Matrix<T> operator*(const Matrix<T>& source) const;

	/**
	 * @fn	Matrix<T> Matrix::operator-(const Matrix<T>& source) const;
	 *
	 * @brief	Subtraction operator.
	 *
	 * @param	source	Source for the.
	 *
	 * @return	The result of the operation.
	 */

	Matrix<T> operator-(const Matrix<T>& source) const;

	/**
	 * @fn	Matrix<T> Matrix::operator+(const Matrix<T>& source) const;
	 *
	 * @brief	Addition operator.
	 *
	 * @param	source	Source for the.
	 *
	 * @return	The result of the operation.
	 */

	Matrix<T> operator+(const Matrix<T>& source) const;

	/**
	 * @fn	Matrix<T>& Matrix::operator=(const Matrix<T>& source);
	 *
	 * @brief	Assignment operator.
	 *
	 * @param	source	Source for the.
	 *
	 * @return	A shallow copy of this instance.
	 */

	Matrix<T>& operator=(const Matrix<T>& source);

	/**
	 * @fn	T& Matrix::operator()(unsigned int rowIndex, unsigned int columnIndex);
	 *
	 * @brief	Function call operator.
	 *
	 * @param	rowIndex   	Zero-based index of the row.
	 * @param	columnIndex	Zero-based index of the column.
	 *
	 * @return	The result of the operation.
	 */

	T& operator()(unsigned int rowIndex, unsigned int columnIndex);

	/**
	 * @fn	T Matrix::operator()(unsigned int rowIndex, unsigned int columnIndex) const;
	 *
	 * @brief	Function call operator.
	 *
	 * @param	rowIndex   	Zero-based index of the row.
	 * @param	columnIndex	Zero-based index of the column.
	 *
	 * @return	The result of the operation.
	 */

	T operator()(unsigned int rowIndex, unsigned int columnIndex) const;

	/**
	 * @typedef	class std::vector<T>::iterator iterator
	 *
	 * @brief	Iterator implementation.
	 */

	typedef class std::vector<T>::iterator iterator;

	/**
	 * @typedef	class std::vector<T>::const_iterator const_iterator
	 *
	 * @brief	Defines an alias representing the constant iterator.
	 */

	typedef class std::vector<T>::const_iterator const_iterator;

	/**
	 * @fn	iterator Matrix::begin();
	 *
	 * @brief	Gets the begin.
	 *
	 * @return	An iterator.
	 */

	iterator begin();

	/**
	 * @fn	const_iterator Matrix::cbegin() const;
	 *
	 * @brief	Gets the cbegin.
	 *
	 * @return	A const_iterator.
	 */

	const_iterator cbegin() const;

	/**
	 * @fn	iterator Matrix::end();
	 *
	 * @brief	Gets the end.
	 *
	 * @return	An iterator.
	 */

	iterator end();

	/**
	 * @fn	const_iterator Matrix::cend() const;
	 *
	 * @brief	Gets the cend.
	 *
	 * @return	A const_iterator.
	 */

	const_iterator cend() const;

	/**
	 * @fn	const_iterator Matrix::begin() const;
	 *
	 * @brief	Gets the begin.
	 *
	 * @return	A const_iterator.
	 */

	const_iterator begin() const;

	/**
	 * @fn	const_iterator Matrix::end() const;
	 *
	 * @brief	Gets the end.
	 *
	 * @return	A const_iterator.
	 */

	const_iterator end() const;

	//------ITERATORS METHOS TO SPEED UP OPERATION (MULTIPLICATION, ADDITION, SUBSTRACTION)

	/**
	 * @fn	iterator Matrix::rowEnd(unsigned int rowNumber);
	 *
	 * @brief	Row end.
	 *
	 * @param	rowNumber	The row number.
	 *
	 * @return	An iterator.
	 */

	iterator rowEnd(unsigned int rowNumber);

	/**
	 * @fn	const_iterator Matrix::rowEnd(unsigned int rowNumber) const;
	 *
	 * @brief	Row end.
	 *
	 * @param	rowNumber	The row number.
	 *
	 * @return	A const_iterator.
	 */

	const_iterator rowEnd(unsigned int rowNumber) const;

	/**
	 * @fn	const_iterator Matrix::rowBegin(unsigned int rowNumber) const;
	 *
	 * @brief	Row begin.
	 *
	 * @param	rowNumber	The row number.
	 *
	 * @return	A row iterator.
	 */

	const_iterator rowBegin(unsigned int rowNumber) const;

	/**
	 * @fn	iterator Matrix::rowBegin(unsigned int rowNumber);
	 *
	 * @brief	Row begin.
	 *
	 * @param	rowNumber	The row number.
	 *
	 * @return	A row iterator.
	 */

	iterator rowBegin(unsigned int rowNumber);

	/**
	 * @fn	inline static bool Matrix::isParallelSet();
	 *
	 * @brief	Query if this instance is parallel set.
	 *
	 * @return	true if parallel set, false if not.
	 */

	inline static bool isParallelSet();

	/**
	 * @fn	static void Matrix::setParallel(bool wantedStatus);
	 *
	 * @brief	Sets a parallel.
	 *
	 * @param	wantedStatus	true to wanted status.
	 */

	static void setParallel(bool wantedStatus);
};

// _____ Constructors

template <typename T>
Matrix<T>::Matrix(const unsigned int rowSize, const unsigned int columSize)
		:_rowSize(rowSize), _columnSize(columSize)
{
	_isMatrixSizeLegit(rowSize, columSize);
	this->_initilizeMatrix();
}

template <typename T>
Matrix<T>::Matrix()
		:Matrix(DEFAULT_MATRIX_ROW_SIZE, DEFAULT_MATRIX_COLUMN_SIZE) {}

template <typename T>
Matrix<T>::Matrix(const Matrix& source):Matrix(source.rows(), source.cols())
{
	*this = source;
}

template <typename T>
Matrix<T>::Matrix(Matrix && source):Matrix(source.rows(), source.cols())
{
	if (this != &source)
	{
		*this = source;
	}
}

template <typename T>
Matrix<T>::Matrix(unsigned int rowSize, unsigned int columnSize, const std::vector<T>& cells)
		:Matrix(rowSize, columnSize)
{
	_isVectorFitsMatrix(cells.size());
	_matrix = cells;
}


//_______Additional General Methods

/** @brief	iniitilize the static variable */
template <typename T>
bool Matrix<T>::s_isParallel = false;

template <typename T>
bool Matrix<T>::isParallelSet()
{
	return s_isParallel;
}

template <typename T>
unsigned int Matrix<T>::_calculateCellNumber(const unsigned int& rowNumber, const unsigned int&
											 columnNumber, const unsigned int& matrixColumns)
{
	return (rowNumber * matrixColumns + columnNumber);
}

//--- Logic Exception Test Methods

template <typename T>
void Matrix<T>::_isMatrixSizeLegit(const unsigned int& rows, const unsigned int& cols) const
{
	if ((rows == 0 && cols != 0) || (rows != 0 && cols == 0))
	{
		throw (std::logic_error(ILLEGAL_MATRIX_SIZE_MSG));
	}
}



template <typename T>
void Matrix<T>::_isIndicesOutOfRange(const unsigned int& row, const unsigned int& column) const
{
	if ((this->_rowSize <= row) || (this->_columnSize <= column))
	{
		throw (std::out_of_range(REQUESTED_INDICES_OUT_BOUND_MSG));
	}
}

template <typename T>
void Matrix<T>::_isVectorFitsMatrix(const unsigned long& vectorSize) const
{
	if (vectorSize != (this->rows() * this->cols()))
	{
		throw (std::out_of_range("Given std::vector size doesn't fit matrix size!"));
	}
}

template <typename T>
void Matrix<T>::_isMatrixSizeEqual(const Matrix& rightMatrix) const
{
	if (this->rows() != rightMatrix.rows() || this->cols() != rightMatrix.cols())
	{
		throw (std::invalid_argument(DIFFRENT_MATRICE_SIZE_MSG));
	}
}

template <typename T>
void Matrix<T>::_isMultipliedMatriceSizeLegit(const Matrix& rightMatrix) const
{
	if (this->cols() != rightMatrix.rows())
	{
		throw (std::invalid_argument(MULTIPLIED_MATRICE_WITH_WRONG_SIZES_MSG));
	}
}

template <typename T>
bool Matrix<T>::isSquareMatrix() const
{
	return (this->rows() == this->cols());
}

//_____Constructor Support Methods

template <typename T>
void Matrix<T>::_initilizeMatrix()
{
	this->_matrix.resize(_rowSize * _columnSize);
}

template <typename T>
void Matrix<T>::_initilizeMatrix(unsigned int rowSize, unsigned int colSize)
{
	this->_rowSize = rowSize;
	this->_columnSize = colSize;
	this->_initilizeMatrix();
}

//___________Genreal Support Methods

template <typename T>
void Matrix<T>::setMatrix(const std::vector<T>& matrix)
{
	this->_matrix = matrix;
}

template <typename T>
void Matrix<T>::setParallel(bool wantedStatus)
{
	std::string parallelStatus = DEFAULT_EMPTY_STRING;
	if (isParallelSet() != wantedStatus)
	{
		if (wantedStatus)
		{
			parallelStatus = PARALLAL_STATUS_LABEL;
		}
		else
		{
			parallelStatus = NON_PARALLEL_STATUS_LABEL;
		}
		std::cout << DEFAULT_PARALLEL_MODE_CHANGE_MSG << std::endl;
		Matrix<T>::s_isParallel = wantedStatus;
	}
}

//__________Operator Implementaion

template <>
inline Matrix<Complex> Matrix<Complex>::trans()
{
	Matrix<Complex> result(cols(), rows());
	for (unsigned int rowIndex = 0; rowIndex < this->rows(); rowIndex ++)
	{
		for (unsigned int columnIndex = 0; columnIndex < this->cols(); columnIndex ++)
		{
			result(columnIndex, rowIndex) = ((*this)(rowIndex, columnIndex)).conj();
		}
	}
	return result;
}

template <typename T>
Matrix<T> Matrix<T>::trans()
{
	Matrix<T> result(cols(), rows());
	for (unsigned int rowIndex = 0; rowIndex < this->rows(); rowIndex ++)
	{
		for (unsigned int columnIndex = 0; columnIndex < this->cols(); columnIndex ++)
		{
			result(columnIndex, rowIndex) = ((*this)(rowIndex, columnIndex));
		}
	}
	return result;
}

template <typename T>
const T Matrix<T>::trace()
{
	T result = DEFAULT_TRACE_RESULT;
	try
	{
		if (! isSquareMatrix())
		{
			throw (std::logic_error(MATRIX_NOT_SQUARE_MSG));
		}

		for (unsigned int rowIndex = 0; rowIndex < this->rows(); rowIndex ++)
		{
			result += (*this)(rowIndex, rowIndex);
		}
	}
	catch (std::exception& exception)
	{
		std::cout << exception.what() << std::endl;
	}

	return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& source)
{
	if (this != &source)
	{
		this->_initilizeMatrix(source.rows(), source.cols());
		this->setMatrix(source.getMatrix());
	}
	return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& source) const
{
	_isMatrixSizeEqual(source);
	Matrix<T> result(this->rows(), this->cols());
	if (! Matrix<T>::isParallelSet())
	{
		_regularAdditionAlgo(source, result);
	}
	else
	{
		_multiThreadedAdditionAlgo(source, result);
	}
	return result;
}
template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& source) const
{
	_isMatrixSizeEqual(source);
	Matrix<T> result(this->_rowSize, this->_columnSize);
	std::transform(this->_matrix.begin(),
	               this->_matrix.end(),
	               source.getMatrix().begin(),
	               result._matrix.begin(),
	               std::minus<T>());
	return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& source) const
{
	_isMultipliedMatriceSizeLegit(source);
	Matrix<T> result(this->rows(), source.cols());
	if (! Matrix<T>::isParallelSet())
	{
		_regularMultAlgo(source, result);
	}
	else
	{
		_multiThreadedMultAlgo(source, result);
	}
	return result;
}

template <typename T>
T Matrix<T>::operator()(unsigned int rowIndex, unsigned int columnIndex) const
{
	_isIndicesOutOfRange(rowIndex, columnIndex);
	return this->_matrix[_calculateCellNumber(rowIndex, columnIndex, this->cols())];
}

template <typename T>
T& Matrix<T>::operator()(unsigned int rowIndex, unsigned int columnIndex)
{
	_isIndicesOutOfRange(rowIndex, columnIndex);
	return this->_matrix[_calculateCellNumber(rowIndex, columnIndex, this->cols())];
}

//---operator support

template <typename T>
void Matrix<T>::_regularMultAlgo(const Matrix<T>& source, Matrix<T>& target) const
{
	std::vector<T>& resultMatrix = target._matrix;

	std::vector<std::vector<T>> leftMatrixColumns(source.cols());
	_vectorColumnSlicer(source, leftMatrixColumns);

	T init = DEFAULT_MATRIX_CELL_VAL;
	for (unsigned int i = 0; i < this->rows(); i ++)
	{
		for (unsigned int j = 0; j < source.cols(); j ++)
		{
			resultMatrix[_calculateCellNumber(i, j, target.cols())] =
					std::inner_product(this->rowBegin(i), this->rowEnd(i),
					                   leftMatrixColumns[j].begin(), init, Matrix::_sumOp,
					                   Matrix::_productOp);
		}
	}
}

template <typename T>
void Matrix<T>::_multiThreadedMultAlgo(const Matrix<T>& source, Matrix<T>& target) const
{
	using std::thread;
	std::vector<thread> rowVectorThreads;
	rowVectorThreads.reserve(this->rows());

	//run and work separatly on each row
	std::vector<T>& resultMatrix = target._matrix;

	std::vector<std::vector<T>> leftMatrixColumns(source.cols());
	_vectorColumnSlicer(source, leftMatrixColumns);

	std::vector<std::vector<T>> rightMatrixRows(this->rows());

	std::vector<std::vector<T>> tempVectorMatrix(_rowSize);

	for (unsigned int i = 0; i < this->rows(); i ++)
	{
		std::vector<T> leftRowVector(this->rowBegin(i), this->rowEnd(i));
		rightMatrixRows[i] = leftRowVector;
		tempVectorMatrix[i].reserve(_columnSize);
		rowVectorThreads.emplace_back([&, i]
		                              {
			                              _multiplyRowByCol(leftMatrixColumns,
			                                                rightMatrixRows[i],
			                                                tempVectorMatrix[i],
			                                                source.cols());
		                              });
	}

	//join threads and put the repush_backsult back to the source
	for (unsigned int j = 0; j < this->_rowSize; j ++)
	{
		rowVectorThreads[j].join();
	}

	for (unsigned int j = 0; j < this->_rowSize; j ++)
	{
		this->_rowToMatrix(tempVectorMatrix[j], resultMatrix, j);
	}
}

template <typename T>
void Matrix<T>::_multiThreadedAdditionAlgo(const Matrix<T>& source, Matrix<T>& target) const
{
	std::vector<std::thread> rowVectorThreads;
	rowVectorThreads.reserve(this->rows());

	for (unsigned int i = 0; i < this->rows(); i ++)
	{
		rowVectorThreads.emplace_back([&, i]
		                              {
			                              _calculateVectorToVectorAdd(*this, source, target,
			                                                               i);
		                              });
	}
	//join threads and put the result back to the matrix
	for (unsigned int j = 0; j < rowVectorThreads.size(); j ++)
	{
		rowVectorThreads[j].join();
	}
}

template <typename T>
void Matrix<T>::_regularAdditionAlgo(const Matrix<T>& source, Matrix<T>& target) const
{
	std::transform(this->_matrix.begin(),
	               this->_matrix.end(),
	               source.getMatrix().begin(),
	               target._matrix.begin(),
	               std::plus<T>());
}

template <typename T>
T Matrix<T>::_sumOp(const T& leftItem, const T& rightItem)
{
	return (leftItem + rightItem);
}

template <typename T>
T Matrix<T>::_productOp(const T& leftItem, const T& rightItem)
{
	return (leftItem * rightItem);
}

template <typename T>
void Matrix<T>::_calculateVectorToVectorAdd(const Matrix<T>& lhs, const Matrix<T>& rhs,
											Matrix<T>& result, const int& rowNum) const
{
	std::transform(lhs.rowBegin(rowNum), lhs.rowEnd(rowNum), rhs.rowBegin(rowNum),
	               result.rowBegin(rowNum), Matrix::_sumOp);
}

template <typename T>
void Matrix<T>::_rowToMatrix(const std::vector<T>& row, std::vector<T>& targetMatrix,
							 const unsigned int rowNumber) const
{
	//first initilize to starting position on the correct row
	unsigned int
			columnCursor = _calculateCellNumber(rowNumber, ROW_COPY_START_POS, this->_columnSize);
	for (unsigned int colIndx = 0; colIndx < _columnSize; colIndx ++)
	{
		targetMatrix[columnCursor + colIndx] = row[colIndx];
	}
}

template <typename T>
void Matrix<T>::_vectorColumnSlicer(const Matrix<T>& source1, std::vector<std::vector<T>>& target)
const
{
	for (unsigned int i = 0; i < source1.cols(); i ++)
	{
		for (unsigned int j = 0; j < source1.rows(); j ++)
		{
			target[i].push_back(source1(j, i));
		}
	}
}

template <typename T>
void Matrix<T>::_multiplyRowByCol(const std::vector<std::vector<T>>& columns, const std::vector<T>&
								  row, std::vector<T>& resultMatrixRow, const unsigned int&
								  numOfLeftMatrixColums) const
{
	T init = DEFAULT_MATRIX_CELL_VAL;
	for (unsigned int i = 0; i < numOfLeftMatrixColums; i ++)
	{
		const std::vector<T>& cursorColumn = columns[i];
		resultMatrixRow[i] = std::inner_product(row.begin(), row.end(), cursorColumn.begin(), init,
		                                        Matrix::_sumOp, Matrix::_productOp);
	}
}
template <typename T>
typename Matrix<T>::iterator Matrix<T>::begin()
{
	return _matrix.begin();
}

template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::cbegin() const
{
	return _matrix.cbegin();
}

template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::cend() const
{
	return _matrix.cend();
}

template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::begin() const
{
	return _matrix.begin();
}

template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::end() const
{
	return _matrix.end();
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::end()
{
	return _matrix.end();
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::rowBegin(unsigned int rowNumber)
{
	return begin() + this->cols() * rowNumber;
}

template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::rowBegin(unsigned int rowNumber) const
{
	return cbegin() + this->cols() * rowNumber;
}

template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::rowEnd(unsigned int rowNumber) const
{
	return rowBegin(rowNumber) + this->cols();
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::rowEnd(unsigned int rowNumber)
{
	return rowBegin(rowNumber) + this->cols();
}

#endif //_MATRIX__HPP

