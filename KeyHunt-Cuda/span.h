#ifndef SPAN_H
#define SPAN_H

#include <cstddef>
#include <iterator>
#include <stdexcept>

// Simple span implementation for C++11 compatibility
// Based on C++20 std::span and GSL guidelines
template <typename T>
class span {
public:
    using element_type = T;
    using value_type = typename std::remove_cv<T>::type;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // Constructors
    constexpr span() noexcept : data_(nullptr), size_(0) {}
    
    constexpr span(pointer ptr, size_type count) : data_(ptr), size_(count) {
        if (ptr == nullptr && count != 0) {
            throw std::runtime_error("span: null pointer with non-zero size");
        }
    }
    
    template <std::size_t N>
    constexpr span(element_type (&arr)[N]) noexcept : data_(arr), size_(N) {}
    
    template <typename Container>
    constexpr span(Container& cont) : data_(cont.data()), size_(cont.size()) {}
    
    // Observers
    constexpr pointer data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return size_; }
    constexpr size_type size_bytes() const noexcept { return size_ * sizeof(element_type); }
    constexpr bool empty() const noexcept { return size_ == 0; }
    
    // Element access
    constexpr reference operator[](size_type idx) const {
        if (idx >= size_) {
            throw std::out_of_range("span: index out of range");
        }
        return data_[idx];
    }
    
    constexpr reference front() const {
        if (empty()) {
            throw std::out_of_range("span: front() called on empty span");
        }
        return data_[0];
    }
    
    constexpr reference back() const {
        if (empty()) {
            throw std::out_of_range("span: back() called on empty span");
        }
        return data_[size_ - 1];
    }
    
    // Iterators
    constexpr iterator begin() const noexcept { return data_; }
    constexpr iterator end() const noexcept { return data_ + size_; }
    constexpr const_iterator cbegin() const noexcept { return data_; }
    constexpr const_iterator cend() const noexcept { return data_ + size_; }
    constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
    constexpr reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
    constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
    constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }
    
    // Subviews
    constexpr span first(size_type count) const {
        if (count > size_) {
            throw std::out_of_range("span: first() count exceeds size");
        }
        return span(data_, count);
    }
    
    constexpr span last(size_type count) const {
        if (count > size_) {
            throw std::out_of_range("span: last() count exceeds size");
        }
        return span(data_ + size_ - count, count);
    }
    
    constexpr span subspan(size_type offset, size_type count = static_cast<size_type>(-1)) const {
        if (offset > size_) {
            throw std::out_of_range("span: subspan() offset exceeds size");
        }
        
        if (count == static_cast<size_type>(-1)) {
            count = size_ - offset;
        }
        
        if (offset + count > size_) {
            throw std::out_of_range("span: subspan() range exceeds size");
        }
        
        return span(data_ + offset, count);
    }
    
    // Conversion to byte span
    template <typename U = std::byte>
    span<const U> as_bytes() const noexcept {
        return span<const U>(reinterpret_cast<const U*>(data_), size_bytes());
    }

private:
    pointer data_;
    size_type size_;
};

// Deduction guides
template <typename T, std::size_t N>
span(T (&)[N]) -> span<T>;

template <typename T>
span(T*, std::size_t) -> span<T>;

template <typename Container>
span(Container&) -> span<typename Container::value_type>;

#endif // SPAN_H