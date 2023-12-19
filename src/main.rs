use rayon::iter::ParallelIterator;

use rand::Rng;
use rayon::iter::IntoParallelIterator;
use sdl2::event::Event;

const MAX_X: u32 = 600;
const MAX_Y: u32 = 400;

const PROGRESSION: u32 = 200;
const TRASPARECNY: u8 = 100;

#[derive(Clone)]
 struct Point{
    x: f32,
    y: f32,
}

impl Point{
    #[allow(unused)]
    const ZERO:Self = Point{x: 0.0, y: 0.0};
    fn normalized(&self) -> Self{
        let mag = self.mag();
        Self {
            x:self.x / mag,
            y: self.y / mag
        }
    }
    fn mag(&self) -> f32{
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

struct Bezier{
    points: Vec<Point>
}



impl Bezier{
    fn new<T: Into<Point>>(points: Vec<T>) -> Self{
        Bezier{
            points: points.into_iter().map(|v|v.into()).collect::<Vec<Point>>()
        }
    }
    fn point(&self, progress: f32) -> Point{
        calc_bezier(&self.points, progress)
    }
}
fn calc_bezier(v: &[Point], progress: f32) -> Point{
    recursive_step(v.len() as u32 - 1, 0, progress, v)
}
fn recursive_step(n: u32, i: u32, progress: f32, v: &[Point]) -> Point{
    if i == n{
        equate(n, i, progress, v)
    }else{
        equate(n, i, progress, v) + recursive_step(n, i + 1, progress, v)
    }
}


fn equate(n: u32, i: u32, progress: f32, v: &[Point]) -> Point{
    v.get(i as usize).unwrap().clone() *
    binomial_iter::BinomialIter::new(n, i).binom() as f32 *
    (1.0 - progress).powi((n - i) as i32) * 
    progress.powi(i as i32)
}

pub fn main() {
    let mut rand = rand::thread_rng();
    let controls = (0..7).into_iter().map(|_| Point::from((rand.gen::<f32>() * MAX_X as f32, rand.gen::<f32>() * MAX_Y as f32))).collect::<Vec<Point>>();
    let mut curve = Bezier::new(controls);


    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
 
    let window = video_subsystem.window("joeboids", MAX_X, MAX_Y)
        .position_centered()
        .build()
        .unwrap();
 
    let mut sdl_canvas = window.into_canvas().accelerated().build().unwrap();
    sdl_canvas.set_blend_mode(sdl2::render::BlendMode::Blend);
    // calculate the points
    let mut points = (0..PROGRESSION).into_par_iter().map(|n|{
        curve.point(n as f32 / PROGRESSION as f32)
    }).collect::<Vec<Point>>();


    let mut velocities = (0..points.len()).map(|_|{
        Point{
            x: rand.gen::<f32>() * 2.0 - 1.0,
            y: rand.gen::<f32>() - 0.5
        }.normalized()
    }).collect::<Vec<Point>>();

    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut should_move = true;
    'running: loop{
        let timer = std::time::Instant::now();
        sdl_canvas.set_draw_color(sdl2::pixels::Color::RGB(0, 0, 0,));
        sdl_canvas.clear();
        sdl_canvas.set_draw_color(sdl2::pixels::Color::RGB(255, 255, 255));
        for event in event_pump.poll_iter(){
            match event{
                Event::Quit { .. } => break 'running,
                Event::KeyDown { keycode: Some(sdl2::keyboard::Keycode::Up), ..} => {
                    if curve.points.len() < 35{
                        curve.points.push(Point{x: rand.gen::<f32>() * MAX_X as f32, y: rand.gen::<f32>() * MAX_Y as f32});
                    }
                },
                Event::KeyDown { keycode: Some(sdl2::keyboard::Keycode::Down), .. } => {
                    if curve.points.len() > 2{
                        curve.points.pop();
                    }
                },
                Event::KeyDown { keycode: Some(sdl2::keyboard::Keycode::Space), .. } => {
                    should_move = !should_move
                }
                _ => ()
             }
        }

        {
            let m = event_pump.mouse_state();
            if m.is_mouse_button_pressed(sdl2::mouse::MouseButton::Left){
                let x = m.x();
                let y = m.y();
                let mut closest = f32::MAX;
                let mut index = 0;
                for (i, point) in curve.points.iter().enumerate(){
                    let dist = ((point.x - x as f32).powi(2) + (point.y - y as f32).powi(2)).sqrt();
                    if dist.abs() < closest{
                        closest = dist;
                        index = i;
                    }
                }
                curve.points[index] = Point{x: x as f32, y: y as f32}

            }
        }
        points = (0..PROGRESSION).into_par_iter().map(|n|{
            curve.point(n as f32 / PROGRESSION as f32)
        }).collect::<Vec<Point>>();
        if should_move{
            curve.points = curve.points.into_iter().zip(velocities.iter_mut()).map(|(p, v)|{
                let x = p.x + v.x;
                let y = p.y + v.y;
                if x > MAX_X as f32 || x < 0.0{
                    v.x = -v.x
                }
                if y > MAX_Y as f32 || y < 0.0{
                    v.y = -v.y
                }
                Point{
                    x,
                    y
                }
            }).collect::<Vec<Point>>();
        }
        let sdl_points = points
            .clone()
            .into_iter()
            .map(|p|sdl2::rect::Point::from(p))
            .collect::<Vec<sdl2::rect::Point>>();
        let _ = sdl_canvas.draw_lines(&*sdl_points);
        for (i, point) in curve.points.iter().enumerate(){
            {
                let max = curve.points.len() as f32;
                sdl_canvas.set_draw_color(sdl2::pixels::Color::RGBA(
                        (((max - i as f32) / max) * 255.0) as u8,   // r
                        ((i as f32 / max) * 255.0) as u8,                   // b
                        0,                                                // g
                        TRASPARECNY.saturating_mul(2)
                    )
                )
            }
            let un = sdl2::rect::Point::new(point.x as i32 - 5, point.y as i32 - 5);
            let four = sdl2::rect::Point::new(point.x as i32 + 5, point.y as i32 + 5);
            let doia = sdl2::rect::Point::new(point.x as i32 - 5, point.y as i32 + 5);
            let toia = sdl2::rect::Point::new(point.x as i32 + 5, point.y as i32 - 5);
            
            let _ = sdl_canvas.draw_line(un, four);
            let _ = sdl_canvas.draw_line(doia, toia);
        }
        let sdl_points = curve.points.clone().into_iter().map(|p|sdl2::rect::Point::from(p)).collect::<Vec<sdl2::rect::Point>>();
        sdl_canvas.set_draw_color(sdl2::pixels::Color::RGBA(0, 255, 255, TRASPARECNY));
        let _ = sdl_canvas.draw_lines(&*sdl_points);

        sdl_canvas.present();
        std::thread::sleep(std::time::Duration::from_millis(10).saturating_sub(timer.elapsed()));
    }
}

impl From<Point> for sdl2::rect::Point{
    fn from(value: Point) -> Self {
        sdl2::rect::Point::new(value.x.round() as i32, value.y.round() as i32)
    }
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
