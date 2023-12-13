// for file handling
use std::fs::File;
use std::io::BufWriter;

use rayon::iter::ParallelIterator;

use rand::Rng;
use rayon::iter::IntoParallelIterator;

const MAX_X: u32 = 50;
const MAX_Y: u32 = 50;

const PROGRESSION: u32 = 5_000;

const START: (f32, f32) = (10.0,10.0);
const END: (f32, f32) = (80.0, 10.0);
const HANDLE_1: (f32, f32) = (900.0, 50.0);
const HANDLE_2: (f32, f32) = (70.0, 70.0);



// for getting command line arguments

// This type needs to hold everything required to draw triangles
// for the BoidCanvas trait
struct PngWriter{
    // 2d 'canvas' that we will draw to
    buffer: Vec<Vec<bool>>,
    // for drawing canvas to a file
    writer: png::Writer<BufWriter<std::fs::File>>,
}

#[derive(Clone)]
struct Point{
    x: f32,
    y: f32,
}

impl Point{
    const ZERO:Self = Point{x: 0.0, y: 0.0};
}

struct Bezier{
    points: Vec<Point>
}



impl Bezier{
    fn new<T: Into<Point>>(points: Vec<T>) -> Self{
        Bezier{
            // points: points.into_iter().map(|p| (p.clone()).into()).collect::<Vec<Point>>()
            points: points.into_iter().map(|v|v.into()).collect::<Vec<Point>>()
        }
    }
    fn point(&self, progress: f32) -> Point{
        calc_bezier(&self.points, progress)
    }
}
fn calc_bezier(v: &[Point], progress: f32) -> Point{
    // 
    // (1.0 - progress).powi(2) * p0 + 2.0 * (1.0 - progress) * progress * p1 + progress.powi(2) * p2
    recursive_step(v.len() as u32 - 1, 0, progress, v)
}
fn recursive_step(n: u32, mut i: u32, progress: f32, v: &[Point]) -> Point{
    if i == n{
         v.get(i as usize).unwrap().clone() * binomial_iter::BinomialIter::new(n, i).binom() as f32 * (1.0 - progress).powi((n - i) as i32) *progress.powi(i as i32)
        // Point{x: 0.0, y: 0.0}
    }else{
        v.get(i as usize).unwrap().clone() * binomial_iter::BinomialIter::new(n, i).binom() as f32 * (1.0 - progress).powi((n - i) as i32) *progress.powi(i as i32) + recursive_step(n, i + 1wn, progress, v)
    }
}

fn equate(n: u32, mut i: u32, progress: f32, v: &[Point]) -> Point{
    v.get(i as usize).unwrap().clone() *
    binomial_iter::BinomialIter::new(n, i).binom() as f32 *
    (1.0 - progress).powi((n - i) as i32) * 
    progress.powi(i as i32)
    + 
    recursive_step(n, i + 1, progress, v)

}

pub fn main() {
    let writer = new_png_writer("bezier.png", MAX_X, MAX_Y);
    let mut canvas = PngWriter{
        writer,
        buffer: new_buffer_bools(MAX_X, MAX_Y)
    };
    let mut rand = rand::thread_rng();
    // let mut points = (0..3).into_iter().map(|_| Point::from((rand.gen::<f32>() * MAX_X as f32, rand.gen::<f32>() * MAX_Y as f32))).collect::<Vec<Point>>();
    let mut points: Vec<Point> = vec![];
    // points.push(Point { x: MAX_X as f32, y: MAX_Y as f32});
    let points = vec![ Point{x: 77.0, y: 40.0}, Point{x: 100.0, y: 100.0}];
    let curve = Bezier::new(points);
    // let mut progress = 0.0;
    // loop{
    //     let _ = draw_point(&mut canvas.buffer, curve.point(progress));
    //     progress += 0.00005;
    //     if progress > 1.0{
    //         break;
    //     }
    // }
    let pixels = (0..PROGRESSION).into_par_iter().map(|n|{
        curve.point(n as f32 / PROGRESSION as f32)
    }).collect::<Vec<Point>>();
    pixels.into_iter().for_each(|p|{
        let _ = draw_point(&mut canvas.buffer, p);
    });
    canvas.write().unwrap();
}


impl From<(f32, f32)> for Point{
    fn from(value: (f32, f32)) -> Self {
        Point { x: value.0, y: value.1 }
    }
}

impl From<Point> for (u32, u32) {
    fn from(value: Point) -> Self {
        (value.x.round() as u32, value.y.round() as u32)
    }
}

impl core::ops::Mul<f32> for Point{
    type Output = Point;
    fn mul(self, rhs: f32) -> Self::Output {
        Self{
            x: self.x * rhs,
            y: self.y * rhs
        }
    }
}
impl core::ops::Add for Point{
    type Output = Point;
    fn add(self, rhs: Self) -> Self::Output {
        Point{
            x: self.x + rhs.x,
            y: self.y + rhs.y
        }
    }
}

impl PngWriter{
    // the draw_triangle function used by the Boid modifies the 
    // buffer, this function will submit the buffer to a file
    fn write(&mut self) -> Result<(), png::EncodingError>{
        self.writer.write_image_data(
            // this function take a Vec<u8>. With a bit depth of
            // one, this is 8 pixels per u8. the width/height won't
            // always be divisible by 8, so there will be 
            // buffer zeros (this is what `vec_u8_from_vec_bool` does)
            self.buffer
                .clone()
                .into_iter()
                .map(|image_row|
                    // these are each a row of out image, buffered with
                    // zeros at the ends
                    vec_u8_from_vec_bool(image_row)
                )
                // we currently have a Vec<Vec<u8>>, we want
                // to flatten that to a Vec<u8>
                .flatten()
                .collect::<Vec<u8>>()
                // required for png::encoder::Writer::write_image_data
                .as_slice()
        )
    }
    
}



// buffer of bools we can 'draw' to
// false represents a dark pixel, true represents a light pixel
fn new_buffer_bools(width: u32, height: u32) -> Vec<Vec<bool>>{
    // for every row
    (0..height).into_iter()
        // map each value (row)
        .map(|_|{
            // which has `width` columns of false
            (0..width).into_iter().map(|_| false)
            // into a Vec<bool>
            .collect::<Vec<bool>>()
    }).collect::<Vec<Vec<bool>>>()
}

// Turns a vec of bools into a vec of u8s, buffering with zeros
// at the end
// ex.
// [   t, t, f, f, t, t, f, f,     t, t,]
// [0b_1__1__0__0__1__1__0__0_, 0b_1__1__000000] <- which is [204, 192]
fn vec_u8_from_vec_bool(v: Vec<bool>) -> Vec<u8>{
    let mut out: Vec<u8> = vec![];
    // we go over the array of bools 8 at a time
    let chunked_bits = v.chunks_exact(8);
    // we save any leftovers for later (if v.len() isn't cleanly divisible by 8)
    let remainder = chunked_bits.remainder();
    // bits is an array of bools
    for bits in chunked_bits{
        let mut byte: u8 = 0;
        // iterate over the bools from back to front
        for (i, bit) in bits.into_iter().rev().enumerate(){
            // 2^0, 2^1, ..., 2^7
            // if bit is false we just... don't add it
            byte += *bit as u8 * 2_u8.pow(i as u32);
        }
        out.push(byte);
    }
    // handle the last one, if present
    if remainder.len() > 0 {
        // same thing as above
        let mut last_byte = 0;
        for (i, bit) in remainder.into_iter().rev().enumerate(){
            // we have to offset the power by how many zeros there has to be (so that the data 
            // is on the leftmost of the last byte)
            last_byte += *bit as u8 * 2_u8.pow(i as u32 + (8 - remainder.len()) as u32);
        }
        out.push(last_byte);
    }

    out
}

fn new_png_writer(path: &str,width: u32, height: u32) -> png::Writer<BufWriter<File>>{
    // open a new file or overwrite an existing one with truncate
    let path = std::path::Path::new(path);
    let file = match std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path){
        Ok(v) => v,
        Err(e) => { match e.kind(){
                std::io::ErrorKind::NotFound{..} => {
                    eprintln!("The path `{}` didn't work. Make sure you've created the directory you specify with -d (it defaults to `output`)", path.display());
                    panic!("{e}");
                },
                _ => panic!("{e}"),
            }
        }
    };
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Grayscale);
    // each pixel is a single bit: dark or light, but they still need to be packaged
    // in u8s because... that's the API
    encoder.set_depth(png::BitDepth::One);
    // after writing the header it's ready to write data
    encoder.write_header().unwrap()
}


fn draw_point<T: Into<(u32, u32)>>(data: &mut Vec<Vec<bool>>, point: T) -> Result<(), ()>{
    edit_point(data, point.into(), true)
}
fn erase_point<T: Into<(u32, u32)>>(data: &mut Vec<Vec<bool>>, point: T) -> Result<(), ()>{
    edit_point(data, point.into(), false)
}
fn edit_point(data: &mut Vec<Vec<bool>>, point: (u32, u32), what: bool) -> Result<(), ()>{
    if let Some(x) = data.get_mut(point.1 as usize){
        if let Some(y) = x.get_mut(point.0 as usize){
            *y = what;
            Ok(())
        }else{Err(())}
    }else{Err(())}
}