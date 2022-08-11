"""
    Base.:+(a::FockConfig, b::TransferConfig)
Add a `FockConfig` to a `TransferConfig` to get a new `FockConfig`
"""
function Base.:(+)(x::FockConfig{N}, y::TransferConfig{N}) where N
    return FockConfig{N}(ntuple(i -> (x[i][1]+y[i][1], x[i][2]+y[i][2]), N))
end
Base.:(+)(x::TransferConfig{N}, y::FockConfig{N}) where N = y + x

"""
    Base.:-(a::FockConfig, b::FockConfig)
Subtract two `FockConfig`'s, returning a `TransferConfig`
"""
function Base.:-(a::FockConfig{N}, b::FockConfig{N}) where N
    return TransferConfig{N}(ntuple(i -> (a[i][1]-b[i][1], a[i][2]-b[i][2]), N))
end

"""
    Base.:-(a::FockConfig, b::TransferConfig)
"""
function Base.:-(a::FockConfig{N}, b::TransferConfig{N}) where N
    return FockConfig{N}(ntuple(i -> (a[i][1]-b[i][1], a[i][2]-b[i][2]), N))
end


